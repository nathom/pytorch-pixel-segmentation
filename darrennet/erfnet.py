import torch
import torch.nn as nn


class non_bottleneck_1d(nn.Module):
    def __init__(self, n_channel, drop_rate, dilated):
        super().__init__()

        self.conv3x1_1 = nn.Conv2d(
            n_channel, n_channel, (3, 1), stride=1, padding=(1, 0), bias=True
        )

        self.conv1x3_1 = nn.Conv2d(
            n_channel, n_channel, (1, 3), stride=1, padding=(0, 1), bias=True
        )

        self.conv3x1_2 = nn.Conv2d(
            n_channel,
            n_channel,
            (3, 1),
            stride=1,
            padding=(1 * dilated, 0),
            bias=True,
            dilation=(dilated, 1),
        )

        self.conv1x3_2 = nn.Conv2d(
            n_channel,
            n_channel,
            (1, 3),
            stride=1,
            padding=(0, 1 * dilated),
            bias=True,
            dilation=(1, dilated),
        )

        self.bn1 = nn.BatchNorm2d(n_channel, eps=1e-03)
        self.bn2 = nn.BatchNorm2d(n_channel, eps=1e-03)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(drop_rate)

    def forward(self, input):
        output = self.conv3x1_1(input)
        output = self.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = self.relu(output)

        output = self.conv3x1_2(output)
        output = self.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if self.dropout.p != 0:
            output = self.dropout(output)

        return self.relu(output + input)  # +input = identity (residual connection)


class DownsamplerBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channel, out_channel - in_channel, (3, 3), stride=2, padding=1, bias=True
        )
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(out_channel, eps=1e-3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn(output)
        return self.relu(output)


class UpsamplerBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channel, out_channel, 3, stride=2, padding=1, output_padding=1, bias=True
        )
        self.bn = nn.BatchNorm2d(out_channel, eps=1e-3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return self.relu(output)


class ERF(nn.Module):
    def __init__(
        self, num_classes: int, input_channels: int, encoder=None
    ):  # use encoder to pass pretrained encoder
        super().__init__()

        self.num_classes = num_classes
        self.input_channels = input_channels

        if encoder is None:
            self.encoder_flag = True
            self.encoder_layers = nn.ModuleList()

            # layer 1, downsampling
            self.initial_block = DownsamplerBlock(self.input_channels, 16)

            # layer 2, downsampling
            self.encoder_layers.append(DownsamplerBlock(in_channel=16, out_channel=64))

            # non-bottleneck 1d - layers 3 to 7
            self.encoder_layers.append(
                non_bottleneck_1d(n_channel=64, drop_rate=0.03, dilated=1)
            )
            self.encoder_layers.append(
                non_bottleneck_1d(n_channel=64, drop_rate=0.03, dilated=1)
            )
            self.encoder_layers.append(
                non_bottleneck_1d(n_channel=64, drop_rate=0.03, dilated=1)
            )
            self.encoder_layers.append(
                non_bottleneck_1d(n_channel=64, drop_rate=0.03, dilated=1)
            )
            self.encoder_layers.append(
                non_bottleneck_1d(n_channel=64, drop_rate=0.03, dilated=1)
            )

            # layer 8, downsampling
            self.encoder_layers.append(DownsamplerBlock(in_channel=64, out_channel=128))

            # non-bottleneck 1d - layers 9 to 16
            self.encoder_layers.append(
                non_bottleneck_1d(n_channel=128, drop_rate=0.3, dilated=2)
            )
            self.encoder_layers.append(
                non_bottleneck_1d(n_channel=128, drop_rate=0.3, dilated=4)
            )
            self.encoder_layers.append(
                non_bottleneck_1d(n_channel=128, drop_rate=0.3, dilated=8)
            )
            self.encoder_layers.append(
                non_bottleneck_1d(n_channel=128, drop_rate=0.3, dilated=16)
            )
            self.encoder_layers.append(
                non_bottleneck_1d(n_channel=128, drop_rate=0.3, dilated=2)
            )
            self.encoder_layers.append(
                non_bottleneck_1d(n_channel=128, drop_rate=0.3, dilated=4)
            )
            self.encoder_layers.append(
                non_bottleneck_1d(n_channel=128, drop_rate=0.3, dilated=8)
            )
            self.encoder_layers.append(
                non_bottleneck_1d(n_channel=128, drop_rate=0.3, dilated=16)
            )

        else:
            self.encoder_flag = False
            self.encoder = encoder

        self.decoder_layers = nn.ModuleList()

        self.decoder_layers.append(UpsamplerBlock(in_channel=128, out_channel=64))
        self.decoder_layers.append(
            non_bottleneck_1d(n_channel=64, drop_rate=0, dilated=1)
        )
        self.decoder_layers.append(
            non_bottleneck_1d(n_channel=64, drop_rate=0, dilated=1)
        )

        self.decoder_layers.append(UpsamplerBlock(in_channel=64, out_channel=16))
        self.decoder_layers.append(
            non_bottleneck_1d(n_channel=16, drop_rate=0, dilated=1)
        )
        self.decoder_layers.append(
            non_bottleneck_1d(n_channel=16, drop_rate=0, dilated=1)
        )

        self.output_conv = nn.ConvTranspose2d(
            in_channels=16,
            out_channels=self.num_classes,
            kernel_size=2,
            stride=2,
            padding=0,
            output_padding=0,
            bias=True,
        )

        # self.activation = nn.Softmax(dim=1)

        # self.apply(weights_init_normal)

    def forward(self, x):
        if self.encoder_flag:
            output = self.initial_block(x)

            for layer in self.encoder_layers:
                output = layer(output)
        else:
            output = self.encoder(x)

        for layer in self.decoder_layers:
            output = layer(output)

        output = self.output_conv(output)
        # output = self.activation(output)

        return output
