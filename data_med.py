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
            # if not (row[0].split("/")[-1]).startswith("IXI"):
            #     continue
            
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
        v_ = volume[mdl, :, :, :].reshape(-1)
        v_ = v_[v_ > 0]  # Use only the brain foreground to calculate the quantile
        p_99 = torch.quantile(v_, 0.99)
        volume[mdl, :, :, :] /= p_99

    return volume*2 - 1

class BrainDataset_3D(Dataset):
    def __init__(self, image_dir, age_file, mode, transform=normalise_percentile):
        if isinstance(image_dir, str):
            image_dir = Path(image_dir)
        images = image_dir.iterdir()
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

        random.seed(10)
        random.shuffle(files)

        n_train = int(len(files) * 0.80)
        self.files = files[:n_train] if mode == "train" else files[n_train :]
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Load NIfTI image
        patient_path = self.files[idx]
        patient_img = nib.load(patient_path['img'])
        patient_img = patient_img.get_fdata()
        age_index = patient_path['age_idx']
        # Convert image to PyTorch tensor
        image = torch.from_numpy(patient_img).float().unsqueeze(0)
        # Apply transform if any
        if self.transform:
            image = self.transform(image)
        
        return image, age_index
       


    
class PatientDataset(torch.utils.data.Dataset):
    """
    Dataset class representing a collection of slices from a single scan.
    """

    def __init__(
        self, patient_dir: Path, process_fun=None, id=None, age_dict=None, age_map=None
    ):
        self.patient_dir = patient_dir
        self.patient_age = int(float(age_dict[patient_dir.name+".nii"]))
        self.patient_age_index = age_map[self.patient_age]
        
        self.slice_paths = sorted(
            list(patient_dir.iterdir()), key=lambda x: int(x.name[6:-4])
        )
        self.process = process_fun
        self.id = id
        self.len = len(self.slice_paths)
        self.idx_map = {x: x for x in range(self.len)}

    def __getitem__(self, idx):
        idx = self.idx_map[idx]
        data = np.load(self.slice_paths[idx])
        if self.process is not None:
            data = self.process(self.id, self.patient_age_index, data['x'])
        return data

    def __len__(self):
        return self.len


class BrainDataset(torch.utils.data.Dataset):
    """
    Dataset class representing a collection of slices from scans from a specific dataset split.
    """

    def __init__(
        self,
        datapath,
        age_file,
        split="train",
        seed=0,
        num_patients=None,
    ):
        self.rng = random.Random(seed)

        assert split in ["train", "test"]
        
        age_dict, age_map = get_age(age_file) # key: age, value: index
        self.split = split

        path = Path(datapath) / f"npy_{split}"

        def process(id, age, x):
            # x: 1, 256, 256
            x_tensor = torch.from_numpy(x[0]).float()
            # center crop to 224
            # x_tensor = x_tensor[:, 16:240, 16:240]
            x_tensor = x_tensor * 2 - 1
            return x_tensor, age, id

        patient_dirs = sorted(list(path.iterdir()))
        self.rng.shuffle(patient_dirs)
        num_patients = len(patient_dirs) if num_patients is None else num_patients

        self.patient_datasets = []
        for i in range(num_patients):
            try :
                self.patient_age = int(float(age_dict[patient_dirs[i].name+".nii"]))
            except KeyError:
                print(f"Image {patient_dirs[i].name} does not have an age label.")
                continue
            
            self.patient_datasets.append(
                PatientDataset(
                    patient_dirs[i], age_dict=age_dict, age_map=age_map, process_fun=process, id=i
                )
            )

        self.dataset = ConcatDataset(self.patient_datasets)

    def __getitem__(self, idx):
        if self.split == "train":
            x, age, _ = self.dataset[idx]  
            return x, age
        else:
            x, age, id = self.dataset[idx] 
            return x, age, id
       
    def __len__(self):
        return len(self.dataset)


# %% load brainage
def load_brainage(
    data_dir,
    age_file,
    split,
    num_patients=None,
):
    assert split in ["train", "test"]

    if split == "train":
        return BrainDataset(
            data_dir,
            age_file,
            split="train",
            num_patients=num_patients,
        )
    else:
        return BrainDataset(
            data_dir,
            age_file,
            split="test",
            num_patients=num_patients,
        )


# %%
def get_brainage_data_iter(
    data_dir,
    age_file,
    batch_size,
    split="train",
    logger=None,
    training=True,
    num_patients=None,
):
    data = load_brainage(data_dir, age_file, split, num_patients=num_patients)

    loader = DataLoader(
        data,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True if split == "train" else False,
        drop_last=True if split == "train" else False,
        sampler=None,
    )

    if logger is not None:
        logger.log(f"data_size: {data.__len__()}")

    training = True if split == "train" else False
    
    if training:
        while True:
            yield from loader
    else:
        yield from loader

if __name__ == "__main__":
    data_dir = Path("/data/amciilab/yiming/DATA/brain_age/extracted/")
    age_dir = Path("/data/amciilab/yiming/DATA/brain_age/masterdata.csv")
    data_set = BrainDataset_3D(
        data_dir, age_dir, mode="train", transform=normalise_percentile
    )
    data_loader = DataLoader(data_set, batch_size=2, shuffle=True)
    # check data
    for i, (x, age) in enumerate(data_loader):
        print(x.shape, age)
        break
    
    # data_dir = Path("/data/amciilab/yiming/DATA/brain_age/preprocessed_data_256_10_IXI")
    # data_loader = get_brainage_data_iter(data_dir=data_dir,age_file=age_dir, batch_size=2, split="train", num_patients=10)
    # x, age = next(data_loader)
    # print(x.shape, age)
