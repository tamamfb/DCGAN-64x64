import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from model import Generator
from torchvision.utils import save_image
from torchmetrics.image.fid import FrechetInceptionDistance
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Z_DIM = 100
CHANNELS_IMG = 3
FEATURES_GEN = 64
IMAGE_SIZE = 64
BATCH_SIZE = 64

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

real_dataset = ImageFolder(root="anime_faces", transform=transform)
real_loader = DataLoader(real_dataset, batch_size=BATCH_SIZE, shuffle=False)

gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
gen.load_state_dict(torch.load("saved_models/generator_epoch_29.pth", map_location=device))
gen.eval()

fid = FrechetInceptionDistance(feature=64).to(device)

for real_batch, _ in real_loader:
    real_batch_uint8 = ((real_batch * 0.5 + 0.5) * 255).clamp(0, 255).to(torch.uint8)
    fid.update(real_batch_uint8.to(device), real=True)


with torch.no_grad():
    for _ in range(len(real_loader)):
        noise = torch.randn(BATCH_SIZE, Z_DIM, 1, 1).to(device)
        fake = gen(noise)
        fake_uint8 = ((fake * 0.5 + 0.5) * 255).clamp(0, 255).to(torch.uint8)
        fid.update(fake_uint8, real=False)

# Compute FID
fid_score = fid.compute()
print(f"FID score: {fid_score.item():.2f}")
