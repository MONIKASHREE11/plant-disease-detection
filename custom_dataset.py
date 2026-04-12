from torch.utils.data import Dataset
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

class CustomImageDataset(Dataset):
    """
    Custom dataset that applies different augmentation strategies
    per disease class using Albumentations.
    """
    def __init__(self, image_paths, labels, label_names, transform_dict):
        self.image_paths = image_paths
        self.labels = labels
        self.label_names = label_names
        self.transform_dict = transform_dict

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.labels[idx]
        label_name = self.label_names[label]

        transform = self.transform_dict.get(label_name, self.transform_dict['default'])
        image = transform(image=image)['image']
        return image, label
