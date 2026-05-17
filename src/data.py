from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class SkinDataset(Dataset):
    def __init__(self, image_dir: Path, *, color_mode: str = "RGBA"):
        image_dir = Path(image_dir)
        self.paths = sorted(
            p for p in image_dir.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )

        if not self.paths:
            raise ValueError(f"No images found in {image_dir}")

        self.color_mode = color_mode
        channels = 4 if color_mode == "RGBA" else 3

        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * channels, [0.5] * channels),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert(self.color_mode)
        return self.transform(image)
