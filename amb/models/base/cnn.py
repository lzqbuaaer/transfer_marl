import torch.nn as nn
from amb.utils.model_utils import init, get_active_func, get_init_method


class CNNLayer(nn.Module):
    def __init__(
        self,
        obs_shape,
        hidden_sizes,
        initialization_method,
        activation_func,
        kernel_size=3,
        stride=1,
    ):
        super(CNNLayer, self).__init__()

        active_func = get_active_func(activation_func)
        init_method = get_init_method(initialization_method)
        gain = nn.init.calculate_gain(activation_func)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        input_channel = obs_shape[0]
        input_width = obs_shape[1]
        input_height = obs_shape[2]

        self.output_size = hidden_sizes[0] // 2 * (input_width - kernel_size + stride) * (input_height - kernel_size + stride)

        layers = [
            init_(
                nn.Conv2d(
                    in_channels=input_channel,
                    out_channels=hidden_sizes[0] // 2,
                    kernel_size=kernel_size,
                    stride=stride,
                )
            ),
            active_func,
            nn.Flatten(),
        ]

        self.cnn = nn.Sequential(*layers)

    def forward(self, x):
        x = x / 255.0
        x = self.cnn(x)
        return x