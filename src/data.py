from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class SkinDataset(Dataset):
    def __init__(self, image_dir: Path, *, color_mode: str = "RGBA", channels: int = 4):
        image_dir = Path(image_dir)
        self.paths = sorted(
            p for p in image_dir.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )

        if not self.paths:
            raise ValueError(f"No images found in {image_dir}")

        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")

        self.color_mode = color_mode
        self.channels = channels
        norm_mean = [0.5] * channels
        norm_std = [0.5] * channels

        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert(self.color_mode)
        return self.transform(image)
