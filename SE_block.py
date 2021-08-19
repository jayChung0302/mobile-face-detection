import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, input_channels, internal_channels):
        super(SEBlock, self).__init__()
        self.down_conv = nn.Conv2d(in_channels=input_channels, \
            out_channels=internal_channels, kernel_size=1, stride=1, bias=True)
        self.up_conv = nn.Conv2d(in_channels=internal_channels, \
            out_channels=input_channels, kernel_size=1, stride=1)
        self.input_channels = input_channels

    def forward(self, input):
        x = F.avg_pool2d(input, kernel_size=input.size(3))
        x = self.down_conv(x)
        x = F.relu(x)
        x = self.up_conv(x)
        x = torch.sigmoid(x)
        x = x * input
        return x

def main():
    module = SEBlock(10, 5)
    input = torch.randn(5,10,224,224)
    output = module(input)
    print(output.size())

if __name__ == '__main__':
    main()

