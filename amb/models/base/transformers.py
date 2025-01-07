import torch.nn as nn
import torch.nn.functional as F
import torch
import argparse

class Attention(nn.Module):
    def __init__(self, emb, heads=8):

        super().__init__()

        self.emb = emb
        self.heads = heads

        self.tokeys = nn.Linear(emb, emb * heads, bias=False)
        self.toqueries = nn.Linear(emb, emb * heads, bias=False)
        self.tovalues = nn.Linear(emb, emb * heads, bias=False)

        self.unifyheads = nn.Linear(heads * emb, emb)

    def forward(self, q, k, v, mask):

        b_q, t_q, e_q = q.size()
        b_k, t_k, e_k = k.size()
        b_v, t_v, e_v = v.size()
        assert (b_q == b_k) and (b_k == b_v)
        assert (t_k == t_v)
        assert (self.emb == e_q) and (self.emb == e_k) and (self.emb == e_v)
        h = self.heads
        keys = self.tokeys(k).view(b_k, t_k, h, e_k)
        queries = self.toqueries(q).view(b_q, t_q, h, e_q)
        values = self.tovalues(v).view(b_v, t_v, h, e_v)

        # compute scaled dot-product self-attention

        # - fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b_k * h, t_k, e_k)
        queries = queries.transpose(1, 2).contiguous().view(b_q * h, t_q, e_q)
        values = values.transpose(1, 2).contiguous().view(b_v * h, t_v, e_v)

        queries = queries / (e_q ** (1 / 4))
        keys = keys / (e_k ** (1 / 4))
        # - Instead of dividing the dot products by sqrt(e), we scale the keys and values.
        #   This should be more memory efficient

        # - get dot product of queries and keys, and scale
        dot = torch.bmm(queries, keys.transpose(1, 2))

        assert dot.size() == (b_q * h, t_q, t_v)

        if mask is not None:
            dot = dot.masked_fill(mask == 0, -1e9)

        dot = F.softmax(dot, dim=2)
        # - dot now has row-wise self-attention probabilities

        # apply the self attention to the values
        out = torch.bmm(dot, values).view(b_q, h, t_q, e_v)

        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b_q, t_q, h * e_v)

        return self.unifyheads(out)

class EncoderBlock(nn.Module):

    def __init__(self, emb, heads, ff_hidden_mult=4, dropout=0.0):
        super().__init__()

        self.attention = Attention(emb, heads=heads)

        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)

        self.ff = nn.Sequential(
            nn.Linear(emb, ff_hidden_mult * emb),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * emb, emb)
        )

        self.do1 = nn.Dropout(dropout)
        self.do2 = nn.Dropout(dropout)

    def forward(self, x_mask):
        x, mask = x_mask

        attended = self.attention(x, x, x, mask)

        x = self.norm1(attended + x)

        x = self.do1(x)

        fedforward = self.ff(x)

        x = self.norm2(fedforward + x)

        x = self.do2(x)

        return x
    
class DecoderBlock(nn.Module):

    def __init__(self, emb, heads, ff_hidden_mult=4, dropout=0.0):
        super().__init__()

        self.self_attention = Attention(emb, heads=heads)
        self.enc_dec_attention = Attention(emb, heads=heads)

        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)
        self.norm3 = nn.LayerNorm(emb)

        self.ff = nn.Sequential(
            nn.Linear(emb, ff_hidden_mult * emb),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * emb, emb)
        )

        self.do1 = nn.Dropout(dropout)
        self.do2 = nn.Dropout(dropout)
        self.do3 = nn.Dropout(dropout)

    def forward(self, x_mask, m_mask):
        x, self_mask = x_mask
        memory, memory_mask = m_mask

        self_attention_output = self.self_attention(x, x, x, self_mask)

        x = self.norm1(self_attention_output + x)

        x = self.do1(x)
        
        enc_dec_attention_output = self.enc_dec_attention(x, memory, memory, memory_mask)
        
        x = self.norm2(enc_dec_attention_output + x)
        
        x = self.do2(x)

        fedforward = self.ff(x)

        x = self.norm3(fedforward + x)

        x = self.do3(x)

        return x


class Encoder(nn.Module):

    def __init__(self, emb, heads, depth):
        super().__init__()
        self.eblocks = nn.ModuleList([EncoderBlock(emb=emb, heads=heads) for _ in range(depth)])

    def forward(self, tokens, h=None, mask=None):
        if h is not None:
            x = torch.cat((tokens, h), 1)
        else:
            x = tokens

        for eblock in self.eblocks:
            x = eblock((x, mask))

        return x
    
class Decoder(nn.Module):

    def __init__(self, emb, heads, depth, output_dim):
        super().__init__()

        self.num_tokens = output_dim

        self.dblocks = nn.ModuleList([DecoderBlock(emb=emb, heads=heads) for _ in range(depth)])

        self.toprobs = nn.Linear(emb, output_dim)
    
    def forward(self, tokens, memory, h=None, mask=None, memory_mask=None):
        if h is not None:
            x = torch.cat((tokens, h), 1)
        else:
            x = tokens

        b, t, e = tokens.size()
        
        for dblock in self.dblocks:
            x = dblock((x, mask), (memory, memory_mask))

        x = self.toprobs(x.view(b * t, e)).view(b, t, self.num_tokens)

        return x
    
class Transformer(nn.Module):

    def __init__(self, emb, heads, depth, output_dim):
        super().__init__()

        self.encoder = Encoder(emb=emb, heads=heads, depth=depth)
        self.decoder = Decoder(emb=emb, heads=heads, depth=depth, output_dim=output_dim)
    
    def forward(self, src, tgt, src_h=None, tgt_h=None, src_mask=None, tgt_mask=None):
        memory = self.encoder(src, h=src_h, mask=src_mask)
        output = self.decoder(tgt, memory, h=tgt_h, mask=tgt_mask, memory_mask=src_mask)

        return output[:, :-tgt_h.shape[1]] if (tgt_h is not None) else output, \
               memory[:, -src_h.shape[1]:] if (src_h is not None) else None, \
               output[:, -tgt_h.shape[1]:] if (tgt_h is not None) else None

def mask_(matrices, maskval=0.0, mask_diagonal=True):

    b, h, w = matrices.size()
    indices = torch.triu_indices(h, w, offset=0 if mask_diagonal else 1)
    matrices[:, indices[0], indices[1]] = maskval


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unit Testing')
    parser.add_argument('--token_dim', default='5', type=int)
    parser.add_argument('--emb', default='32', type=int)
    parser.add_argument('--heads', default='3', type=int)
    parser.add_argument('--depth', default='2', type=int)
    parser.add_argument('--ally_num', default='5', type=int)
    parser.add_argument('--enemy_num', default='5', type=int)
    parser.add_argument('--episode', default='20', type=int)
    args = parser.parse_args()


    # testing the agent
    agent = Encoder(None, args).cuda()
    hidden_state = agent.init_hidden().cuda().expand(args.ally_num, 1, -1)
    tensor = torch.rand(args.ally_num, args.ally_num+args.enemy_num, args.token_dim).cuda()
    q_list = []
    for _ in range(args.episode):
        q, hidden_state = agent.forward(tensor, hidden_state, args.ally_num, args.enemy_num)
        q_list.append(q)
