import torch.nn as nn
from torchvision.models.resnet import ResNet34_Weights, resnet34


class FCN(nn.Module):
    def __init__(self, n_class, resnet_backbone=False):
        super().__init__()
        self.n_class = n_class
        self.resnet = resnet_backbone

        # Encoder
        if not resnet_backbone:
            self.resnet = False
            self.conv1 = nn.Conv2d(
                3, 32, kernel_size=(3, 3), stride=2, padding=1, dilation=1
            )
            self.bnd1 = nn.BatchNorm2d(32)
            self.conv2 = nn.Conv2d(
                32, 64, kernel_size=(3, 3), stride=2, padding=1, dilation=1
            )
            self.bnd2 = nn.BatchNorm2d(64)
            self.conv3 = nn.Conv2d(
                64, 128, kernel_size=(3, 3), stride=2, padding=1, dilation=1
            )
            self.bnd3 = nn.BatchNorm2d(128)
            self.conv4 = nn.Conv2d(
                128, 256, kernel_size=(3, 3), stride=2, padding=1, dilation=1
            )
            self.bnd4 = nn.BatchNorm2d(256)
            self.conv5 = nn.Conv2d(
                256, 512, kernel_size=(3, 3), stride=2, padding=1, dilation=1
            )
            self.bnd5 = nn.BatchNorm2d(512)
            self.relu = nn.ReLU(inplace=True)
        else:
            self.resnet = True
            self.relu = nn.ReLU(inplace=True)
            self.resnet = True
            model = resnet34(
                weights=ResNet34_Weights.DEFAULT, norm_layer=nn.BatchNorm2d
            )

            # encoder
            self.initial = list(model.children())[:4]
            self.initial = nn.Sequential(*self.initial)
            self.layer1 = model.layer1
            self.layer2 = model.layer2
            self.layer3 = model.layer3
            self.layer4 = model.layer4

        # Decoder
        self.deconv1 = nn.ConvTranspose2d(
            512,
            512,
            kernel_size=(3, 3),
            stride=2,
            padding=1,
            dilation=1,
            output_padding=1,
        )
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(
            512,
            256,
            kernel_size=(3, 3),
            stride=2,
            padding=1,
            dilation=1,
            output_padding=1,
        )
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=(3, 3),
            stride=2,
            padding=1,
            dilation=1,
            output_padding=1,
        )
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=(3, 3),
            stride=2,
            padding=1,
            dilation=1,
            output_padding=1,
        )
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(
            64,
            32,
            kernel_size=(3, 3),
            stride=2,
            padding=1,
            dilation=1,
            output_padding=1,
        )
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, self.n_class, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Encoder forward
        if not self.resnet:
            x1 = self.bnd1(self.relu(self.conv1(x)))  # Only provided line
            x2 = self.bnd2(self.relu(self.conv2(x1)))
            x3 = self.bnd3(self.relu(self.conv3(x2)))
            x4 = self.bnd4(self.relu(self.conv4(x3)))
            x5 = self.bnd5(self.relu(self.conv5(x4)))
        else:
            x1 = self.layer1(self.initial(x))
            x2 = self.layer2(x1)
            x3 = self.layer3(x2)
            x5 = self.layer4(x3)

        # Decoder forward
        y1 = self.bn1(self.relu(self.deconv1(x5)))  # Only provided line
        y2 = self.bn2(self.relu(self.deconv2(y1)))
        y3 = self.bn3(self.relu(self.deconv3(y2)))
        y4 = self.bn4(self.relu(self.deconv4(y3)))
        y5 = self.bn5(self.relu(self.deconv5(y4)))

        score = self.classifier(y5)  # (16, 21, 224, 224)
        output = self.softmax(score)

        # return score  # size=(batch size, n_class, H of image, W of image)
        return output  # size=(batch size, n_class, H of image, W of image)
