import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np


class CIFARDataset(Dataset):

    def __init__(self, x, y, transforms=None):

        self.x = x
        self.y = y
        self.transforms = transforms  # albumentations.Compose

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        
        image = self.x[idx] 
        if self.transforms is not None:
            image = self.transforms(image=image)['image']
        image = np.transpose(image, (2, 0, 1))
        x = torch.from_numpy(image.astype(np.float32))  # [c, h, w]
        y = self.y[idx]
        
        return x, y

    
def get_loader(x, y, batch_size=32, num_workers=0, transforms=None, shuffle=False):
    dataset = CIFARDataset(x, y, transforms=transforms)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
    return loader