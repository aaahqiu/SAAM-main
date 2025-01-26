from torch.utils.data import Dataset
import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import cv2
import matplotlib.pyplot as plt

class XKDataset(Dataset):
    def __init__(self, root_path, domain='train', image_size=(256, 256)):
        super(XKDataset, self).__init__()
        self.root_path = root_path
        self.image_size = image_size
        self.domain = domain
        self.image_dir = os.path.join(self.root_path, 'images', domain)
        self.label_dir = os.path.join(self.root_path, 'masks', domain)
        self.name_list = os.listdir(self.image_dir)

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        batch_input = {}
        image = cv2.imread(os.path.join(self.image_dir, self.name_list[index]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = Image.open(os.path.join(self.label_dir, self.name_list[index]))
        mask = np.array(mask)

        if self.domain == 'train':
            image, mask = self._flip(image, mask)
            image = Image.fromarray(np.uint8(image))
            image = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.ColorJitter(brightness=[0.9,1.4], contrast=[0.6,2.2], saturation=0, hue=0),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])(image)
        else:
            image = Image.fromarray(image)
            image = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])(image)
        mask = torch.tensor(mask)

        mask = mask.squeeze()
        mask = mask.long()

        batch_input['image'] = image
        batch_input['label'] = mask
        batch_input['name'] = self.name_list[index]

        return batch_input

    def _flip(self, image, mask):
        if np.random.rand() > 0.5:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()
        if np.random.rand() > 0.5:
            image = np.flipud(image).copy()
            mask = np.flipud(mask).copy()
        return image, mask

    def _rotate(self, image, mask):
        if np.random.rand() > 0.5:
            angle = np.random.randint(-180, 180)
            image = Image.fromarray(image)
            mask = Image.fromarray(mask)
            image = image.rotate(angle)
            mask = mask.rotate(angle)
            image = np.array(image)
            mask = np.array(mask)
        return image, mask



if __name__ == '__main__':
    dataset = XKDataset(root_path='/data/',domain='train',image_size=(1024,1024))
    batch_input = dataset[105]
    image = batch_input['image']
    mask = batch_input['label']
    name = batch_input['name']

    # visualize image
    image = image.numpy()
    image = np.transpose(image, (1, 2, 0))
    plt.imshow(image)
    plt.show()

