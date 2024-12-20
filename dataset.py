import os
import torch
import numpy as np
from PIL import Image
from typing import Optional, Tuple, List
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class JesterDataset(Dataset):
    """
    PyTorch Dataset class for the Jester hand gesture recognition dataset.
    Optimized for use with TimeSFormer architecture.
    """
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        num_frames: int = 16,
        frame_size: Tuple[int, int] = (224, 224),
        normalize: bool = True,
        temporal_sample: str = 'uniform'
    ):
        """
        Initialize the dataset.

        Args:
            root_dir (str): Path to the Jester dataset root directory
            split (str): Dataset split ('train', 'val', or 'test')
            num_frames (int): Number of frames to sample from each video
            frame_size (tuple): Size to resize frames to (height, width)
            normalize (bool): Whether to normalize pixel values
            temporal_sample (str): Sampling strategy ('uniform' or 'random')
        """
        self.root_dir = root_dir
        self.videos_dir = os.path.join(root_dir, '20bn-jester-v1')  # Updated video directory
        self.split = split
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.temporal_sample = temporal_sample

        # Updated file names to match actual dataset structure
        split_map = {
            'train': 'jester-v1-train.csv',
            'val': 'jester-v1-validation.csv',
            'test': 'jester-v1-test.csv'
        }

        self.labels_file = os.path.join(root_dir, split_map[split])
        self.label_map_file = os.path.join(root_dir, 'jester-v1-labels.csv')

        # Load label map first
        self.label_map = self._load_label_map()
        # Then load annotations which depend on label map
        self.videos, self.labels = self._load_annotations()

        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize(frame_size),
            transforms.ToTensor()
        ])

        if normalize:
            self.transform = transforms.Compose([
                self.transform,
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
            ])

    def _load_label_map(self) -> dict:
        """Load mapping between label names and indices."""
        label_map = {}
        with open(self.label_map_file, 'r') as f:
            for idx, line in enumerate(f):
                label_map[line.strip()] = idx
        return label_map

    def _get_label_index(self, label_name: str) -> int:
        """Convert label name to index using the label map."""
        return self.label_map.get(label_name, -1)

    def _load_annotations(self) -> Tuple[List[str], List[int]]:
        """Load video paths and their corresponding labels."""
        videos, labels = [], []

        with open(self.labels_file, 'r') as f:
            for line in f:
                parts = line.strip().split(';')
                if len(parts) == 2:
                    video_id, label_name = parts
                    videos.append(video_id)
                    if self.split != 'test':
                        label_idx = self._get_label_index(label_name)
                        labels.append(label_idx)
                    else:
                        labels.append(-1)

        return videos, labels

    def _load_video(self, video_path: str) -> List[Image.Image]:
        """Load all frames from a video directory."""
        frames = []
        try:
            frame_paths = sorted(os.listdir(video_path))

            for frame_file in frame_paths:
                if frame_file.endswith('.jpg'):
                    frame_path = os.path.join(video_path, frame_file)
                    frame = Image.open(frame_path).convert('RGB')
                    frames.append(frame)

            if not frames:
                raise ValueError(f"No frames found in {video_path}")

        except Exception as e:
            print(f"Error loading video from {video_path}: {str(e)}")
            raise

        return frames

    def _sample_frames(self, frames: List[Image.Image]) -> List[Image.Image]:
        """Sample the required number of frames."""
        num_frames = len(frames)

        if self.temporal_sample == 'uniform':
            indices = np.linspace(0, num_frames - 1, self.num_frames, dtype=int)
        else:
            indices = sorted(np.random.choice(
                num_frames, self.num_frames, replace=num_frames < self.num_frames
            ))

        return [frames[i] for i in indices]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a video and its label.

        Returns:
            tuple: (video_tensor, label)
                - video_tensor: shape (C, T, H, W)
                - label: integer label
        """
        video_id = self.videos[idx]
        video_path = os.path.join(self.videos_dir, video_id)
        frames = self._load_video(video_path)
        frames = self._sample_frames(frames)

        # Apply transforms and stack frames
        frames = torch.stack([self.transform(frame) for frame in frames])
        # Reshape to (C, T, H, W) as expected by TimeSFormer
        frames = frames.permute(1, 0, 2, 3)

        return frames, self.labels[idx]

    def __len__(self) -> int:
        """Return the number of videos in the dataset."""
        return len(self.videos)

    @property
    def num_classes(self) -> int:
        """Return the number of classes in the dataset."""
        return len(self.label_map)


def test_jester_dataset():
    train_dataset = JesterDataset(
        root_dir='jester_dataset',
        split='train',
        num_frames=16,
        frame_size=(224, 224)
    )

    # Create a DataLoader with batch_size=2
    test_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)

    # Get one batch
    videos, labels = next(iter(test_dataloader))

    # Print information about the batch
    print(f"Dataset size: {len(train_dataset)} videos")
    print(f"Number of classes: {train_dataset.num_classes}")
    print(f"\nBatch information:")
    print(f"Video tensor shape: {videos.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Labels: {labels}")
    print(f"\nVideo tensor statistics:")
    print(f"Min value: {videos.min():.3f}")
    print(f"Max value: {videos.max():.3f}")
    print(f"Mean value: {videos.mean():.3f}")


if __name__ == '__main__':
    exit(test_jester_dataset())
