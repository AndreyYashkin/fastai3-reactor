import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, nz):
        super(Generator, self).__init__()
        self.process_noise = nn.Sequential(
            nn.Linear(nz, 7*7*256),
            nn.BatchNorm1d(7*7*256),
            nn.LeakyReLU()
            )

        self.cnn = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            # bs, 128, 7, 7
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            # bs, 64, 14, 14
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.process_noise(x)
        return self.cnn(x.view(-1, 256, 7, 7))

class Discriminator(nn.Module):
    def __init__(self, dropout_p=0):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # (bs, 1, 28, 28)
            nn.Conv2d(1, 64, 4, stride=2, padding=3),
            nn.Dropout(dropout_p),
            nn.LeakyReLU(),
            # (bs, 64, 16, 16)
            nn.Conv2d(64, 128, 4, stride=2, padding=3),
            nn.Dropout(dropout_p),
            nn.LeakyReLU(),
            # (bs, 128, 10, 10)
            nn.Flatten(),
            nn.Linear(128*10*10, 1)
        )
    
    def forward(self, input):
        output = self.main(input)
        return output#.view(-1) 

class ConditionalGenerator(nn.Module):
    def __init__(self, nz):
        super(ConditionalGenerator, self).__init__()
        self.label_embedding = nn.Embedding(10, 10)

        self.process_noise = nn.Sequential(
            nn.Linear(nz+10, 7*7*256),
            nn.BatchNorm1d(7*7*256),
            nn.LeakyReLU()
            )

        self.cnn = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            # bs, 128, 7, 7
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            # bs, 64, 14, 14
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1, bias=False),
            nn.Tanh()
            )

    def forward(self, x, labels):
        labels.__class__ = torch.Tensor # HACK to avoid Tensor subclassing issues
        gen_input = torch.cat((x, self.label_embedding(labels)), -1)
        x = self.process_noise(gen_input)
        return self.cnn(x.view(-1, 256, 7, 7))

class ConditionalDiscriminator(nn.Module):
    def __init__(self, dropout_p=0):
        super(ConditionalDiscriminator, self).__init__()
        self.label_embedding = nn.Embedding(10, 1*28*28)
        self.main = nn.Sequential(
            # (bs, 2, 28, 28)
            nn.Conv2d(2, 64, 4, stride=2, padding=3),
            nn.Dropout(dropout_p),
            nn.LeakyReLU(),
            # (bs, 64, 16, 16)
            nn.Conv2d(64, 128, 4, stride=2, padding=3),
            nn.Dropout(dropout_p),
            nn.LeakyReLU(),
            # (bs, 128, 10, 10)
            nn.Flatten(),
            nn.Linear(128*10*10, 1)
        )
    
    def forward(self, img, labels):
        labels.__class__ = torch.Tensor # HACK to avoid Tensor subclassing issues
        # Concatenate label embedding and image to produce input
        emb = self.label_embedding(labels).view(-1, 1, 28, 28)
        input = torch.cat((img, emb), 1)
        output = self.main(input)
        return output#.view(-1) 
