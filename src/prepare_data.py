from pathlib import Path
from torchvision.datasets import FashionMNIST
from torchvision import transforms

ROOT = Path("data/FashionMNIST")
ROOT.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    # Unduh train dan test; transformasi dilakukan di pipeline training.
    FashionMNIST(root="data", train=True, download=True, transform=transforms.ToTensor())
    FashionMNIST(root="data", train=False, download=True, transform=transforms.ToTensor())
    print("Dataset Fashion-MNIST tersimpan di data/FashionMNIST.")
    print("Selanjutnya jalankan: dvc init, lalu dvc add data/FashionMNIST")
