import sys 
import os
sys.path.append(os.path.realpath('./'))

from pathlib import Path
import csv
import random
import torch
from torchvision.utils import make_grid, save_image
import numpy as np

from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
import nibabel as nib
from skimage.util import view_as_blocks


def get_age(file_path):
    data = {}
    with open(file_path, "r") as csvfile:
        csv_reader = csv.reader(
            csvfile,
            skipinitialspace=True,
        )
        for row in csv_reader:
            if row[0] == "patientid":
                continue
            # IXI only
            if not (row[0].split("/")[-1]).startswith("IXI"):
                continue
            
            data[row[0].split("/")[-1]] = int(float(row[1]))

    age_map = {k: v for v, k in enumerate(set(data.values()))} # age:index
    # stats count age frequency
    age_freq = {}
    for age in data.values():
        age_freq[age] = age_freq.get(age, 0) + 1
    
    return data, age_map, age_freq

def normalise_percentile(volume):
    """
    Normalise the intensity values in each modality by scaling by 99 percentile foreground (nonzero) value.
    """
    for mdl in range(volume.shape[0]):
        v_ = volume[mdl, ...].reshape(-1)
        v_ = v_[v_ > 0]  # Use only the brain foreground to calculate the quantile
        p_99 = torch.quantile(v_, 0.99)
        volume[mdl, ...] /= p_99
    return volume*2 - 1

def load_patient(idx, files):
    patient = files[idx]
    patient_img = nib.load(patient['img'])
    patient_img = patient_img.get_fdata()
    patient_img = patient_img[16:240, 16:240, 16:240] # crop the volume to 224 
    patient_img = torch.from_numpy(patient_img).float().unsqueeze(0) # 1, 224, 224, 224
    age_index = patient['age_idx']
    name = patient['name']
    return patient_img, age_index, name

class BrainDataset_3D(Dataset):
    def __init__(self, image_dir, age_file, mode, transform=normalise_percentile):
        if isinstance(image_dir, str):
            image_dir = Path(image_dir)
        # images = image_dir.iterdir()
        
        assert mode in ["train", "val"]
        
        if mode == "train":
            image_dir = image_dir / "train"
        elif mode == "val":
            image_dir = image_dir / "val"
            
        images = sorted(list(image_dir.glob('IXI*')))
        # images = sorted(list(image_dir.iterdir()))
        age_dict, age_map, age_freq = get_age(age_file)
        self.mode = mode
        self.transform = transform
        files = []
        for img in images:
            try:
                age = int(float(age_dict[img.name]))
                files.append({"img": img, "age_idx": age_map[age], 'name': img.name})
            except KeyError:
                print(f"Image {img.name} does not have an age label.")

        self.files = files
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Load NIfTI image 
        patient_img, age_index, name = load_patient(idx, self.files)
        # Apply transform if any
        if self.transform:
            image = self.transform(patient_img)
        return image, age_index, name
    
    
class BrainDataset_3D_Patch_Single(Dataset):
    def __init__(self, files, id, transform):
        patient_img, self.age_index, self.name = load_patient(id, files) # 1, 224, 224, 224
        if transform:
            patient_img = transform(patient_img) 
            
        patient_blocks = view_as_blocks(patient_img[0].numpy(), block_shape=(32, 32, 32)) # 7, 7, 7, 32, 32, 32
        self.patient_blocks = patient_blocks.reshape(1, -1, 32, 32, 32) # 1, n_patches, 32, 32, 32
        
    def __len__(self):
        return self.patient_blocks.shape[1]
    
    def __getitem__(self, idx):
        block = torch.from_numpy(self.patient_blocks[0:1, idx, ...])
        return block, self.age_index, self.name

class BrainDataset_3D_Patch(BrainDataset_3D):
    def __init__(self, image_dir, age_file, mode, transform=normalise_percentile):
        super().__init__(image_dir, age_file, mode, transform)
        self.patient_datasets = [BrainDataset_3D_Patch_Single(self.files, i, transform) for i in range(len(self.files))]
        self.dataset = ConcatDataset(self.patient_datasets)
        
    def __getitem__(self, idx):
        image_blocks, age_index, name = self.dataset[idx]
        return image_blocks, age_index, name
    
    def __len__(self):
        return len(self.dataset)
    
           
class BrainDataset_2D_Single(Dataset):
    def __init__(self, files, id, transform):
        
        patient_img, self.age_index, self.name = load_patient(id, files) # 1, 224, 224, 224
        if transform:
            patient_img = transform(patient_img)
        # 1, 224, 224
        self.slices = [patient_img[..., i] for i in range(patient_img.shape[-1])] # from the last slices, shape 1, 256, 256
       
    def __getitem__(self, idx):
        image = self.slices[idx]
        return image, self.age_index, self.name
    
    def __len__(self):
        return len(self.slices)

class BrainDataset_2D(BrainDataset_3D):
    def __init__(self, image_dir, age_file, mode, transform=normalise_percentile):
        super().__init__(image_dir, age_file, mode, transform)
        self.patient_datasets = [BrainDataset_2D_Single(self.files, i, transform) for i in range(len(self.files))]
        self.dataset = ConcatDataset(self.patient_datasets)
        
    def __getitem__(self, idx):
        image_slices, age_index, name = self.dataset[idx]
        return image_slices, age_index, name
    
    def __len__(self):
        return len(self.dataset)
   

if __name__ == "__main__":
    # data_dir = Path("/data/amciilab/yiming/DATA/brain_age/extracted/")
    # age_dir = Path("/data/amciilab/yiming/DATA/brain_age/masterdata.csv")
    
    data_dir = Path("/data/amciilab/Data_yiming")
    age_dir = Path("/data/amciilab/Data_yiming/masterdata.csv")
    
    _, age_map, age_freq = get_age(age_dir) # age:index
    reversed_age_map = {v: k for k, v in age_map.items()} # index:age

    # let's start with 2D dataset, which the top view of 3D MRI
    data_set = BrainDataset_2D(
        data_dir, age_dir, mode="train", transform=normalise_percentile
    ) # for validation, change mode to "val"
    
    # Than, 3D dataset
    # data_set = BrainDataset_3D(
    #     data_dir, age_dir, mode="train", transform=normalise_percentile
    # ) 
    
    data_loader = DataLoader(data_set, batch_size=2, shuffle=True)
    # check data
    for i, (x, age_index, name) in enumerate(data_loader):
        print(f'image_shape: {x.shape}')
        # use reversed_age_map to get the age by its index
        age1 = reversed_age_map[age_index[0].item()]
        age2 = reversed_age_map[age_index[1].item()]
        print(f'age1: {age1}, age2: {age2}')
        print(f'image_min: {x.min()}, image_max: {x.max()}')
        print(f'patient_name: {name}')
        print(f'# of data: {data_set.__len__()}')
        break
