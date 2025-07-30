import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch

class DAVIS2017Dataset(Dataset):
    def __init__(self, root_dir,
                       split='train', resolution='480p',
                       transform=None, return_mask=True,
                       vid_len=25, overlap=0):
        """
        Args:
            root_dir (str): Path to DAVIS dataset root.
            split (str): 'train', 'val', or 'test-dev' (uses a text file listing videos).
            resolution (str): '480p' or 'full-resolution'.
            transform (callable, optional): Transform to apply to images and masks.
            return_mask (bool): Whether to return ground truth masks.
        """
        self.root_dir = root_dir
        self.resolution = resolution
        self.return_mask = return_mask
        self.transform = transform

        self.vid_len = vid_len
        assert vid_len > 0, "Video length must be greater than 0."
        assert overlap < vid_len, "Overlap must be less than video length."

        split_file = os.path.join(root_dir, 'ImageSets/2017', f'{split}.txt')
        with open(split_file, 'r') as f:
            self.video_list = [line.strip() for line in f.readlines()]

        self.samples = []
        for vid in self.video_list:
            frame_dir = os.path.join(root_dir, 'JPEGImages', resolution, vid)
            mask_dir = os.path.join(root_dir, 'Annotations', resolution, vid) if return_mask else None
            frame_files = sorted(os.listdir(frame_dir))
            for frame_idx in range(0, len(frame_files)-vid_len+1, vid_len-overlap):
                # frame_path = os.path.join(frame_dir, frame_files[frame_idx])
                # mask_path = os.path.join(mask_dir, frame_files[frame_idx].replace('.jpg', '.png')) if return_mask else None
                self.samples.append((frame_dir, mask_dir, frame_files[frame_idx:frame_idx+vid_len]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
       
        frame_dir, mask_dir, frame_files = self.samples[idx]
        images = []
        masks = []

        for frame_file in frame_files:
            frame_path = os.path.join(frame_dir, frame_file)
            if self.return_mask:
                mask_path = os.path.join(mask_dir, frame_file.replace('.jpg', '.png'))
            image = Image.open(frame_path).convert('RGB')
            if self.return_mask:
                mask = Image.open(mask_path)

            if self.transform:
                if self.return_mask:
                    mask = self.transform(mask)
                image = self.transform(image)
            images.append(image)
            if self.return_mask:
                masks.append(mask)
        
        images = torch.stack(images)
        if self.return_mask:
            masks = torch.stack(masks)
        return (images, masks) if self.return_mask else images