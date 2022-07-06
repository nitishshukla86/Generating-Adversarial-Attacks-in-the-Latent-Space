import torch
import torch.nn as nn
import torchvision

class Generator(nn.Module):
    def __init__(self, nc, ngf,ndf, embedding_dim):
        super(Generator, self).__init__()
        self.nc = nc
        self.ngf = ngf
        self.ndf = ndf
        self.embedding_dim = embedding_dim
        self.decoder = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.embedding_dim, self.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            # state size. (ngf * 8) x 4 x 4
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            # state size. (ngf * 4) x 8 x 8
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            # state size. (ngf * 2) x 16 x 16
            nn.ConvTranspose2d(self.ngf * 2, self.ngf * 1, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 1),
            nn.ReLU(True),
            # state size. (ngf * 1) x 32 x 32
            nn.ConvTranspose2d(self.ngf * 1, self.nc, 4, 2, 1, bias=False),
            nn.Tanh(),
            # state size. (nc) x 64 x 64
        )
        self.encoder=nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(self.nc, self.ndf * 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf * 2) x 32 x 32
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf * 4) x 16 x 16
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf * 4) x 8 x 8
            nn.Conv2d(self.ndf * 8, self.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf * 8) x 4 x 4
            nn.Conv2d(self.ndf * 8, self.embedding_dim, 4, 1, 0, bias=False),
            # state size. (embedding_dim) x 1 x 1
        )
    def forward(self, input):
        output = self.encoder(input)
        return self.decoder(output)
    
    





def Discriminator(nc):
    model=torchvision.models.resnet18()
    if nc==1:
        model.conv1=nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc=nn.Linear(512,10)
    return model