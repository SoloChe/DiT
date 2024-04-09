# import monai.data as md
# from monai import transforms
# from monai.utils import first

from pathlib import Path
import csv
import random
import torch
from torchvision.utils import make_grid, save_image
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
import nibabel as nib

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
    patient_img = torch.from_numpy(patient_img).float().unsqueeze(0) # 1, 256, 256, 256
    age_index = patient['age_idx']
    return patient_img, age_index 

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
        age_dict, age_map, age_freq = get_age(age_file)
        self.mode = mode
        self.transform = transform
        files = []
        for img in images:
            try:
                age = int(float(age_dict[img.name]))
                files.append({"img": img, "age_idx": age_map[age]})
            except KeyError:
                print(f"Image {img.name} does not have an age label.")

        self.files = files
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Load NIfTI image 
        patient_img, age_index = load_patient(idx, self.files)
        # Apply transform if any
        if self.transform:
            image = self.transform(patient_img)
        # crop the volume to 224
        image = image[:, 16:240, 16:240, 16:240]
        return image, age_index
           
class BrainDataset_2D_Single(Dataset):
    def __init__(self, files, id, transform):
        
        patient_img, self.age_index = load_patient(id, files)
        if transform:
            patient_img = transform(patient_img)
        # crop the volume to 224
        patient_img = patient_img[:, 16:240, 16:240, 16:240]
         
        self.slices = [patient_img[..., i] for i in range(patient_img.shape[-1])] # from the last slices, shape 1, 256, 256
       
    def __getitem__(self, idx):
        image = self.slices[idx]
        return image, self.age_index
    
    def __len__(self):
        return len(self.slices)

class BrainDataset_2D(BrainDataset_3D):
    def __init__(self, image_dir, age_file, mode, transform=normalise_percentile):
        super().__init__(image_dir, age_file, mode, transform)
        self.patient_datasets = [BrainDataset_2D_Single(self.files, i, transform) for i in range(len(self.files))]
        self.dataset = ConcatDataset(self.patient_datasets)
        
    def __getitem__(self, idx):
        image_slices, age_index = self.dataset[idx]
        return image_slices, age_index
    
    def __len__(self):
        return len(self.dataset)
   

if __name__ == "__main__":
    data_dir = Path("/data/amciilab/yiming/DATA/brain_age/extracted/")
    age_dir = Path("/data/amciilab/yiming/DATA/brain_age/masterdata.csv")
    data_set = BrainDataset_2D(
        data_dir, age_dir, mode="train", transform=normalise_percentile
    )
    data_loader = DataLoader(data_set, batch_size=64, shuffle=True)
    # check data
    for i, (x, age) in enumerate(data_loader):
        print(x.shape, age)
        print(x.min(), x.max())
        print(data_set.__len__())
        break
    
    # data_dir = Path("/data/amciilab/yiming/DATA/brain_age/preprocessed_data_256_10_IXI")
    # data_loader = get_brainage_data_iter(data_dir=data_dir,age_file=age_dir, batch_size=2, split="train", num_patients=10)
    # x, age = next(data_loader)
    # print(x.shape, age)
