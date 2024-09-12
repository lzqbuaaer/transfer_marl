import torch as th
import torch.nn as nn


class VDNMixer(nn.Module):
    def __init__(self):
        super(VDNMixer, self).__init__()

    def forward(self, agent_qs, batch):
        bs = agent_qs.size(0)
        q_tot = th.sum(agent_qs, dim=1, keepdim=True)
        q_tot = q_tot.view(bs, -1, 1) # [B, 1, 1]
        return q_tot