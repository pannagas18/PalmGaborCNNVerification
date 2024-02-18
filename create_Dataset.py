import os, glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class PalmprintTrainDataset_TripletMarginLoss(Dataset):
    """returns training dataset structured for PalmprintDataset_TripletMarginLoss loss"""
    def __init__(self, img_dir, image_extension = ".bmp", transform=None):
        self.img_dir = img_dir
        self.image_extension = image_extension
        self.transform = transform
        self._path = os.path.join(self.img_dir, f"*{self.image_extension}")
        self.all_images = sorted(glob.glob(pathname=self._path))
        self.group_examples()
    
    def __len__(self):
        return len(glob.glob(self._path))

    
    def group_examples(self):

        slots =np.arange(1,len(self.all_images), 10)
        self.grouped_examples = {}
        for i in range(1,len(self.all_images)//10):
            self.grouped_examples[i] = np.arange(slots[i-1], slots[i])

    def __getitem__(self, index):

        # FUTURE => use index to select an anchor and then select P and N accordingly

        # select a random class
        selected_class = np.random.randint(1,len(self.all_images)//10)

        # select a random_index for anchor
        random_index_A = np.random.randint(self.grouped_examples[selected_class][0], self.grouped_examples[selected_class][-1])

        # select a random_index for positive example
        random_index_P = np.random.randint(self.grouped_examples[selected_class][0], self.grouped_examples[selected_class][-1])

        # make sure same image is not picked
        while random_index_A == random_index_P:
            random_index_P = np.random.randint(self.grouped_examples[selected_class][0], self.grouped_examples[selected_class][-1])

        # select negative example
        other_selected_class = np.random.randint(1,len(self.all_images)//10)

        # make sure same class is not selected
        while other_selected_class == selected_class:
            other_selected_class = np.random.randint(1,len(self.all_images)//10)
        
        random_index_N = np.random.randint(self.grouped_examples[other_selected_class][0], self.grouped_examples[other_selected_class][-1])
        
        img_path_A = os.path.join(self.img_dir, str(random_index_A).zfill(5)+self.image_extension)
        img_path_P = os.path.join(self.img_dir, str(random_index_P).zfill(5)+self.image_extension)
        img_path_N = os.path.join(self.img_dir, str(random_index_N).zfill(5)+self.image_extension)
        

        img_A = Image.open(img_path_A)
        img_P = Image.open(img_path_P)
        img_N = Image.open(img_path_N)

        if self.transform:
            img_A = self.transform(img_A)
            img_P = self.transform(img_P)
            img_N = self.transform(img_N)
        
        return img_A, img_P, img_N




class PalmprintTestDataset_TripletMarginLoss(Dataset):
    """returns testing dataset structured for PalmprintDataset_TripletMarginLoss loss"""
    def __init__(self, img_dir, image_extension = ".bmp", transform=None, target_transform=None):
        self.img_dir = img_dir
        self.image_extension = image_extension
        self.transform = transform
        self.target_transform = target_transform
        self._path = os.path.join(self.img_dir, f"*{self.image_extension}")
        self.all_images = sorted(glob.glob(pathname=self._path))
        self.group_examples()
    
    def __len__(self):
        return len(glob.glob(self._path))
    
    
    def group_examples(self):

        slots =np.arange(1,len(self.all_images), 10)
        self.grouped_examples = {}
        for i in range(1,len(self.all_images)//10):
            self.grouped_examples[i] = np.arange(slots[i-1], slots[i])

    def __getitem__(self, index):

        # select a random class
        selected_class = np.random.randint(1,len(self.all_images)//10) # 600 SHOULD BE (LENGHT OF THE TESTING DATA/10)

        # select a random_index for first image
        random_index_1 = np.random.randint(self.grouped_examples[selected_class][0], self.grouped_examples[selected_class][-1])

        if index % 2 == 0: # select positive example (both images are from same class)
            random_index_2 = np.random.randint(self.grouped_examples[selected_class][0], self.grouped_examples[selected_class][-1])

            # make sure same image is not picked
            while random_index_1 == random_index_2:
                random_index_2 = np.random.randint(self.grouped_examples[selected_class][0], self.grouped_examples[selected_class][-1])

            # positive sample will have distance = 0
            target = torch.tensor(0, dtype=torch.float)

        else: # select negative example (img1 and img2 are from different class)
            other_selected_class = np.random.randint(1,len(self.all_images)//10)

            # make sure same class is not selected
            while other_selected_class == selected_class:
                other_selected_class = np.random.randint(1,len(self.all_images)//10)
            
            random_index_2 = np.random.randint(self.grouped_examples[other_selected_class][0], self.grouped_examples[other_selected_class][-1])

            # negative sample will have distance = 1
            target = torch.tensor(1, dtype=torch.float)
        
        img_1_path = os.path.join(self.img_dir, str(random_index_1).zfill(5)+self.image_extension)
        img_2_path = os.path.join(self.img_dir, str(random_index_2).zfill(5)+self.image_extension)
        

        img_1 = Image.open(img_1_path)
        img_2 = Image.open(img_2_path)

        if self.transform:
            img_1 = self.transform(img_1)
            img_2 = self.transform(img_2)
        if self.target_transform:
            target = self.target_transform(target)
        
        return img_1, img_2, target
