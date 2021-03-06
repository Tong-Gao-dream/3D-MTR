import torch


class _G(torch.nn.Module):
    def __init__(self, args):
        super(_G, self).__init__()
        self.args = args
        self.cube_len = 32

        padd = (0, 0, 0)
        if self.cube_len == 32:
            padd = (1, 1, 1)

        self.layer1 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(768, self.cube_len * 8, kernel_size=4, stride=2, bias=args.bias,
                                     padding=padd),
            torch.nn.BatchNorm3d(self.cube_len * 8),
            torch.nn.ReLU()
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.cube_len * 8, self.cube_len * 4, kernel_size=4, stride=2, bias=args.bias,
                                     padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(self.cube_len * 4),
            torch.nn.ReLU()
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.cube_len * 4, self.cube_len * 2, kernel_size=4, stride=2, bias=args.bias,
                                     padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(self.cube_len * 2),
            torch.nn.ReLU()
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.cube_len * 2, self.cube_len, kernel_size=4, stride=2, bias=args.bias,
                                     padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(self.cube_len),
            torch.nn.ReLU()
        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.cube_len, 1, kernel_size=4, stride=2, bias=args.bias, padding=(1, 1, 1)),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        out = x.view(-1, 768, 1, 1, 1)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)

        return out
