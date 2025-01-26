import torch.nn as nn
import torch

class ResnetGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        return x


if __name__ == '__main__':
    x = torch.randn(4, 3, 1024, 1024)
    model = ResnetGenerator()
    # out = model(x)
    # print(out.shape)
