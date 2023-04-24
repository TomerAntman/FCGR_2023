#
# Created on March 2022
#
# Copyright (c) 2022 Meitar Ronen
#

import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer

def transform_embeddings(transform_type, data):
    if transform_type == "normalize":
        return torch.Tensor(Normalizer().fit_transform(data))
    elif transform_type == "min_max":
        return torch.Tensor(MinMaxScaler().fit_transform(data))
    elif transform_type == "standard":
        return torch.Tensor(StandardScaler().fit_transform(data))
    elif transform_type == "divide_by_max":
        # each image in the batch is divided by its max value
        # the dimension of the data is (batch_size, feature_dim)
        return data / torch.max(data, dim=1, keepdim=True)[0]

    else:
        raise NotImplementedError()

class MyDataset:
    @property
    def data_dim(self):
        """
        The dimension of the loaded data
        """
        return self._data_dim

    def __init__(self, args):
        self.args = args
        self.data_dir = args.dir

    def get_train_data(self):
        raise NotImplementedError()

    def get_validation_data(self):
        raise NotImplementedError()

    def original_dim(self):
        raise NotImplementedError()

    def get_train_loader(self):
        train_loader = torch.utils.data.DataLoader(
            self.get_train_data(),
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=6,
        )
        return train_loader

    def get_validation_loader(self):
        validation_data = self.get_validation_data()
        if len(validation_data) > 0:     
            return torch.utils.data.DataLoader(validation_data, batch_size=self.args.batch_size, shuffle=False, num_workers=6)
        else:
            return None

    def get_loaders(self):
        return self.get_train_loader(), self.get_validation_loader()

    def get_original_dim(self):
        return self.original_dim()


class TensorDatasetWrapper(TensorDataset):
    def __init__(self, data, labels):
        super().__init__(data, labels)
        self.data = data
        self.targets = labels

class CustomDataset(MyDataset):
    def __init__(self, args):
        super().__init__(args)
        self.transformer = transforms.Compose([transforms.ToTensor()])
        self._data_dim = 32*32*3

    
    def get_data(self, train_validation = 'train'):
        try:
            # the data name is "train.npy" or "validation.npy". load it as numpy array and then convert to torch tensor
            codes = np.load(os.path.join(self.data_dir, f"{train_validation}_data.npy"))

            #codes = torch.load(os.path.join(self.data_dir, f"{train_validation}_data.pt"))
        except FileNotFoundError:
            if train_validation=='validation':
                print(f"validation data not found! running only with train data")
                return TensorDatasetWrapper(torch.empty(0), torch.empty(0))
            else:
                raise Exception(f"Training data file not found in path {self.data_dir}")

        if type(codes).__module__ == np.__name__ : # if the data is numpy array
            codes = torch.from_numpy(codes)
        if codes.dim()==3: # [B, H, W] -> [B, H, W, C]
            codes = torch.unsqueeze(codes, dim=-1)
        self.args.input_dim = codes.size()
        if codes.dim() > 2:
            # the channel dimension is the minimal one if there are 4 dimensions. If there are 3 dimensions, then there is one channel
            self.args.is_image = True
            self.args.n_channels = 1 if codes.dim() == 3 else min(codes.size()[1:])
            # flatten
            self.original_shape = codes.size()[1:]
            codes = codes.view(codes.size()[0],-1)
            # if later you want to unflatten a specific tensor, use:
            # tensor.view(original_shape)

        if self.args.transform_input_data:
            codes = transform_embeddings(self.args.transform_input_data, codes)
        if self.args.use_labels_for_eval:
            # labels are txt file like "train_labels.txt" or "validation_labels.txt" and the labels are integers separated by new line. load it as numpy array and then convert to torch tensor
            labels = np.loadtxt(os.path.join(self.data_dir, f"{train_validation}_labels.txt")).astype(int)
            labels = torch.from_numpy(labels)
            # labels = torch.Tensor(torch.load(os.path.join(self.data_dir, f"{train_validation}_labels.pt")))
            # labels = torch.Tensor(labels).to(torch.int)

        else:
            labels = torch.zeros((codes.size()[0]))
        if train_validation == 'train':
            self._data_dim = codes.size()[1]
        codes = codes.to(torch.float32)
        return TensorDatasetWrapper(codes, labels)
    
    def get_train_data(self):
        return self.get_data('train')

    def get_validation_data(self):
        return self.get_data('validation')

    def original_dim(self):
        return self.original_shape

    # def get_name_labels(self, labels):
    #     # The folder that contained the numpy files of the codes and labels also contains a file called "integer_labels_dict.txt"
    #     # This file contains the mapping between the integer labels and the string labels
    #     #labels = self.targets.numpy()
    #     with open(os.path.join(self.data_dir, "integer_labels_dict.txt"), "r") as f:
    #         name_dict = f.read()
    #     name_dict = eval(name_dict)
    #     # the keys are the full names and the values are the integer labels
    #     # switch the keys and values:
    #     name_dict = {v: k for k, v in name_dict.items()}
    #     name_labels = [name_dict[label] for label in labels]
    #     return name_labels


def merge_datasets(set_1, set_2):
    """
    Merged two TensorDatasets into one
    """
    merged = torch.utils.data.ConcatDataset([set_1, set_2])
    return merged


def generate_mock_dataset(dim, len=3, dtype=torch.float32):
    """Generates a mock TensorDataset

    Args:
        dim (tuple): shape of the sample
        len (int): number of samples. Defaults to 10.
    """
    # Make sure train and validation set are of the same type
    if type(dim) == int:
        data = torch.rand((len, dim))
    else:
        data = torch.rand((len, *dim))
    data = torch.tensor(data.clone().detach(), dtype=dtype)
    return TensorDataset(data, torch.zeros(len))
