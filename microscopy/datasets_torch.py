import numpy as np
import os

import torch
from torch.utils.data import Dataset
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from torchvision import transforms

from .datasets_utils import extract_random_patches_from_image, extract_random_patches_from_folder

#####################################
#
# Functions that will be used to define and create the Pytorch dataset

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        hr, lr = sample["hr"], sample["lr"]
        # Pytorch is (batch, channels, width, height)
        hr = hr.transpose((2, 0, 1))
        lr = lr.transpose((2, 0, 1))
        return {"hr": torch.from_numpy(hr), "lr": torch.from_numpy(lr)}

class RandomHorizontalFlip(object):
    """Random horizontal flip"""
    def __init__(self):
        self.rng = np.random.default_rng()

    def __call__(self, sample):
        hr, lr = sample["hr"], sample["lr"]

        if self.rng.random() < 0.5:
            hr = np.flip(hr, 1)
            lr = np.flip(lr, 1)

        return {"hr": hr.copy(), "lr": lr.copy()}

class RandomVerticalFlip(object):
    """Random vertical flip"""
    def __init__(self):
        self.rng = np.random.default_rng()

    def __call__(self, sample):
        hr, lr = sample["hr"], sample["lr"]

        if self.rng.random() < 0.5:
            hr = np.flip(hr, 0)
            lr = np.flip(lr, 0)

        return {"hr": hr.copy(), "lr": lr.copy()}

class RandomRotate(object):
    """Random rotation"""
    def __init__(self):
        self.rng = np.random.default_rng()

    def __call__(self, sample):
        hr, lr = sample["hr"], sample["lr"]

        k = self.rng.integers(4)

        hr = np.rot90(hr, k=k)
        lr = np.rot90(lr, k=k)

        return {"hr": hr.copy(), "lr": lr.copy()}


class PytorchDataset(Dataset):
    """Pytorch's Dataset type object used to obtain the train and
    validation information during the training process. Saves the
    filenames as an attribute and only loads the ones rquired for
    the training batch, reducing the required RAM memory during
    and after the training.
    """

    def __init__(
        self,
        hr_data_path,
        lr_data_path,
        filenames,
        scale_factor,
        crappifier_name,
        lr_patch_shape,
        transformations,
        datagen_sampling_pdf,
        val_split=None,
        validation=False,
        verbose=0
    ):
        self.hr_data_path = hr_data_path
        self.lr_data_path = lr_data_path

        if val_split is None:
            self.filenames = filenames
        elif validation:
            self.filenames = filenames[: int(val_split * len(filenames))]
        else:
            self.filenames = filenames[int(val_split * len(filenames)) :]

        self.transformations = transformations
        self.scale_factor = scale_factor
        self.crappifier_name = crappifier_name
        self.lr_patch_shape = lr_patch_shape

        self.datagen_sampling_pdf = datagen_sampling_pdf

        # In order to not calculate the actual scale factor on each step, its calculated on the initialization  
        _, _, actual_scale_factor = extract_random_patches_from_folder(
                                        self.hr_data_path,
                                        self.lr_data_path,
                                        [self.filenames[0]],
                                        scale_factor=self.scale_factor,
                                        crappifier_name=self.crappifier_name,
                                        lr_patch_shape=self.lr_patch_shape,
                                        datagen_sampling_pdf=self.datagen_sampling_pdf,
                                    )
        self.actual_scale_factor = actual_scale_factor

        self.verbose = verbose

    def __len__(self):
        """
        Returns the length of the object.
        Which will be used for the number of images on each epoch (not the batches).

        :return: int
            The length of the object.
        """
        return len(self.filenames)

    def __getitem__(self, idx):
        """
        Returns a sample from the dataset.

        Parameters:
            - idx (int): The index of the sample to retrieve.

        Returns:
            - sample (dict): A dictionary containing the high-resolution and low-resolution patches of an image.
                - hr (ndarray): The high-resolution patches.
                - lr (ndarray): The low-resolution patches.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        hr_image_path = os.path.join(self.hr_data_path, self.filenames[idx])
        if self.lr_data_path is not None:
            lr_image_path = os.path.join(self.lr_data_path, self.filenames[idx])
        else:
            lr_image_path = None

        aux_lr_patches, aux_hr_patches = extract_random_patches_from_image(
            hr_image_path,
            lr_image_path,
            self.actual_scale_factor,
            self.crappifier_name,
            self.lr_patch_shape,
            self.datagen_sampling_pdf,
            verbose=self.verbose
        )

        lr_patches = np.expand_dims(aux_lr_patches, axis=-1)
        hr_patches = np.expand_dims(aux_hr_patches, axis=-1)

        sample = {"hr": hr_patches, "lr": lr_patches}

        if self.transformations:
            sample = self.transformations(sample)

        if self.verbose > 3:
            print('__get_item__')
            print(sample)

        return sample

class PytorchDataModuler(pl.LightningDataModule):
    def __init__(
            self, 
            lr_patch_size_x: int = 128,
            lr_patch_size_y: int = 128,
            batch_size: int = 8,
            scale_factor: int = 2,
            datagen_sampling_pdf: int = 1,
            rotation: bool = True,
            horizontal_flip: bool = True,
            vertical_flip: bool = True,
            train_hr_path: str = "",
            train_lr_path: str = "",
            train_filenames: list = [],
            val_hr_path: str = "",
            val_lr_path: str = "",
            val_filenames: list = [],
            test_hr_path: str = "",
            test_lr_path: str = "",
            test_filenames: list = [],
            crappifier_method: str = "downsampleonly",
            verbose: int = 0,
            ):
        #Define required parameters here
        super().__init__()

        self.lr_patch_size_x = lr_patch_size_x 
        self.lr_patch_size_y = lr_patch_size_y
        self.batch_size = batch_size
        self.scale_factor = scale_factor
        self.datagen_sampling_pdf = datagen_sampling_pdf

        self.rotation = rotation
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip

        self.train_hr_path = train_hr_path
        self.train_lr_path = train_lr_path
        self.train_filenames = train_filenames

        self.val_hr_path = val_hr_path
        self.val_lr_path = val_lr_path
        self.val_filenames = val_filenames

        self.test_hr_path = test_hr_path
        self.test_lr_path = test_lr_path
        self.test_filenames = test_filenames

        self.crappifier_method = crappifier_method

        self.verbose = verbose

    
    def prepare_data(self):
        # Define steps that should be done
        # on only one GPU, like getting data.
        pass
    
    def setup(self, stage=None):
        # Define steps that should be done on 
        # every GPU, like splitting data, applying
        # transform etc.

        print(f'Dataset setup stage: {stage}')

        # Assign Train/val split(s) for use in Dataloaders
        if stage == "fit":
            train_transformations = []

            if self.horizontal_flip:
                train_transformations.append(RandomHorizontalFlip())
            if self.vertical_flip:
                train_transformations.append(RandomVerticalFlip())
            if self.rotation:
                train_transformations.append(RandomRotate())

            train_transformations.append(ToTensor())

            train_transf = transforms.Compose(train_transformations)
            val_transf = ToTensor()
            
            if self.val_hr_path is None:
                self.train_dataset = PytorchDataset(
                    hr_data_path=self.train_hr_path,
                    lr_data_path=self.train_lr_path,
                    filenames=self.train_filenames,
                    scale_factor=self.scale_factor,
                    crappifier_name=self.crappifier_method,
                    lr_patch_shape=(
                        self.lr_patch_size_x,
                        self.lr_patch_size_y,
                    ),
                    transformations=train_transf,
                    datagen_sampling_pdf=self.datagen_sampling_pdf,
                    val_split=0.1,
                    validation=False,
                    verbose=self.verbose
                )

                self.val_dataset = PytorchDataset(
                    hr_data_path=self.train_hr_path,
                    lr_data_path=self.train_lr_path,
                    filenames=self.train_filenames,
                    scale_factor=self.scale_factor,
                    crappifier_name=self.crappifier_method,
                    lr_patch_shape=(
                        self.lr_patch_size_x,
                        self.lr_patch_size_y,
                    ),
                    transformations=val_transf,
                    datagen_sampling_pdf=self.datagen_sampling_pdf,
                    val_split=0.1,
                    validation=True,
                    verbose=self.verbose
                )

            else:
                self.train_dataset = PytorchDataset(
                    hr_data_path=self.train_hr_path,
                    lr_data_path=self.train_lr_path,
                    filenames=self.train_filenames,
                    scale_factor=self.scale_factor,
                    crappifier_name=self.crappifier_method,
                    lr_patch_shape=(
                        self.lr_patch_size_x,
                        self.lr_patch_size_y,
                    ),
                    transformations=train_transf,
                    datagen_sampling_pdf=self.datagen_sampling_pdf,
                    verbose=self.verbose
                )
                
                self.val_dataset = PytorchDataset(
                    hr_data_path=self.val_hr_path,
                    lr_data_path=self.val_lr_path,
                    filenames=self.val_filenames,
                    scale_factor=self.scale_factor,
                    crappifier_name=self.crappifier_method,
                    lr_patch_shape=(
                        self.lr_patch_size_x,
                        self.lr_patch_size_y,
                    ),
                    transformations=val_transf,
                    datagen_sampling_pdf=self.datagen_sampling_pdf,
                    verbose=self.verbose
                )

        if stage == "test":        
            self.test_dataset = PytorchDataset(
                hr_data_path=self.test_hr_path,
                lr_data_path=self.test_lr_path,
                filenames=self.test_filenames,
                scale_factor=self.scale_factor,
                crappifier_name=self.crappifier_method,
                lr_patch_shape=None,
                transformations=ToTensor(),
                datagen_sampling_pdf=self.datagen_sampling_pdf,
            )

        if stage == "predict":        
            # Is the same as the test_dataset but it also needs to be defined
            self.predict_dataset = PytorchDataset(
                hr_data_path=self.test_hr_path,
                lr_data_path=self.test_lr_path,
                filenames=self.test_filenames,
                scale_factor=self.scale_factor,
                crappifier_name=self.crappifier_method,
                lr_patch_shape=None,
                transformations=ToTensor(),
                datagen_sampling_pdf=self.datagen_sampling_pdf,
            )
    
    def train_dataloader(self):
        # Return DataLoader for Training Data here
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=32)
    
    def val_dataloader(self):
        # Return DataLoader for Validation Data here
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=32)
    
    def test_dataloader(self):
        # Return DataLoader for Testing Data here
        return DataLoader(self.test_dataset, batch_size=1, num_workers=32)

    def predict_dataloader(self):
        # Return DataLoader for Predicting Data here        
        return DataLoader(self.predict_dataset, batch_size=1, num_workers=32)

    def teardown(self, stage):
        if stage == "fit":
            del self.train_dataset
            del self.val_dataset
        if stage == "test":        
            del self.test_dataset
        if stage == "predict":        
            del self.predict_dataset
#
#####################################