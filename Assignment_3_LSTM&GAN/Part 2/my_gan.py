import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
from datetime import datetime

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets

# Generator Model using Transposed Convolutions
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.init_size = 7  # Starting size after upsampling
        self.fc = nn.Linear(latent_dim, 128 * self.init_size * self.init_size)
        
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 1, kernel_size=7, stride=1, padding=3),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.fc(z)
        out = out.view(out.size(0), 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


# Discriminator Model using Convolutions
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        validity = self.model(img)
        return validity


# Weight Initialization
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0)
    elif classname.find("Linear") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0)


# Training Function
def train(dataloader, discriminator, generator, optimizer_G, optimizer_D):
    adversarial_loss = nn.BCELoss()
    
    for epoch in range(args.n_epochs):
        for i, (imgs, _) in enumerate(dataloader):
            batch_size = imgs.size(0)
            valid = torch.ones((batch_size, 1), requires_grad=False).cuda()
            fake = torch.zeros((batch_size, 1), requires_grad=False).cuda()

            imgs = imgs.cuda()

            # Train Discriminator multiple times
            for _ in range(args.d_steps):
                optimizer_D.zero_grad()
                real_loss = adversarial_loss(discriminator(imgs), valid)
                z = torch.randn(batch_size, args.latent_dim).cuda()
                gen_imgs = generator(z)
                fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()
            z = torch.randn(batch_size, args.latent_dim).cuda()
            gen_imgs = generator(z)
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)
            g_loss.backward()
            optimizer_G.step()

            # Print progress
            if i % 100 == 0:
                print(
                    f"[Epoch {epoch}/{args.n_epochs}] [Batch {i}/{len(dataloader)}] "
                    f"[D loss: {d_loss.item():.6f}] [G loss: {g_loss.item():.6f}]"
                )

            # Save Images
            if (epoch * len(dataloader) + i) % args.save_interval == 0:
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                save_image(gen_imgs[:25],
                           f'images/{epoch * len(dataloader) + i}.png',
                           nrow=5, normalize=True)

# Main Function
def main():
    os.makedirs('images', exist_ok=True)

    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                       ])),
        batch_size=args.batch_size, shuffle=True)

    generator = Generator(args.latent_dim).cuda()
    discriminator = Discriminator().cuda()
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    train(dataloader, discriminator, generator, optimizer_G, optimizer_D)
    torch.save(generator.state_dict(), "mnist_generator.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
    parser.add_argument('--save_interval', type=int, default=500, help='save every SAVE_INTERVAL iterations')
    parser.add_argument('--d_steps', type=int, default=5, help='number of discriminator steps per generator step')
    args = parser.parse_args()

    main()
