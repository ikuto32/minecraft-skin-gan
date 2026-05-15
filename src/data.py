from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class SkinDataset(Dataset):
    def __init__(self, image_dir: Path):
        self.paths = sorted(
            p for p in image_dir.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )

        if not self.paths:
            raise ValueError(f"No images found in {image_dir}")

        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert("RGBA")
        return self.transform(image)
