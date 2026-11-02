"""Air-hockey adapter using the LanguageTable recipe's four clips/episode."""
import random
import torch
from torch.utils.data import Dataset
from datasets.airhockey_h5_store import EpisodeStore


class AirHockeyH5Dataset(Dataset):
    def __init__(self, root, mode, sample_length=21, image_size=128):
        self.image_size = (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)
        if self.image_size not in ((128, 128), (72, 128)):
            raise ValueError("Supported shapes: native 128x128 or table crop 72x128")
        split = "valid" if mode in ("val", "valid", "validation") else mode
        if split not in ("train", "valid"):
            raise ValueError("This dataset has train and valid splits, no test split")
        self.store = EpisodeStore(root, split)
        self.mode = split
        self.sample_length = int(sample_length)
        self.clips_per_video = 4 if split == "train" else 1
        if any(e["length"] < self.sample_length for e in self.store.episodes):
            raise ValueError("Episode shorter than requested clip")

    def __len__(self):
        return len(self.store.episodes) * self.clips_per_video

    def locate(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode, segment = divmod(index, self.clips_per_video)
        if self.mode == "valid":
            return episode, 0
        # Stratify valid clip starts across the entire 1200-frame episode.
        count = self.store.episodes[episode]["length"] - self.sample_length + 1
        lo = segment * count // self.clips_per_video
        hi = max(lo, (segment + 1) * count // self.clips_per_video - 1)
        return episode, random.randint(lo, hi)

    def __getitem__(self, index):
        episode, start = self.locate(index)
        images, actions = self.store.read(episode, start, self.sample_length)
        if self.image_size == (72, 128):
            images = images[:, 28:100, :, :].copy()
        # LPWM uses action[t] for frame t -> t+1: the stored alignment is correct.
        video = torch.from_numpy(images).permute(0, 3, 1, 2).float().div_(255)
        return video, torch.from_numpy(actions), "", torch.empty(0), torch.ones(self.sample_length, dtype=torch.int64)
