import numpy as np
import nrrd
import os
import scipy.ndimage
from glob import glob
import torch
from torch.utils import data

from . import utils


class Dataset(data.Dataset):
    
    """
    Class describing a dataset for lung segmentation
    Attributes:
        - list_scans: List of CT scans in the dataset
        - data_path: path to directory containing scans and masks (preprocessing output)
        - mode: 2d will return slices
        - scan_size: size of CT-scans
        - n_classes: number of output class
    """

    def __init__(self, list_scans, data_path, mode="3d", scan_size=[128, 256, 256], n_classes=1):
        self.list_scans = list_scans
        self.data_path = data_path
        self.mode = mode
        self.scan_size = scan_size
        self.n_classes = n_classes

    def __len__(self):
        return len(self.list_scans)

    def __getitem__(self, index, verbose=False):
        # load scan and mask
        if verbose == True:
            print("Loading", self.list_scans[index])
        scan = self.list_scans[index]
        scan_path = os.path.join(self.data_path, scan + ".npy")
        mask_path = os.path.join(self.data_path, scan + "_mask.npy")
        ct_scan = np.load(scan_path)
        seg_mask = np.load(mask_path)

        if self.mode == "2d":
            return ct_scan[:, np.newaxis, :], seg_mask[:, np.newaxis, :]
        else:
            return ct_scan[np.newaxis, :], seg_mask[np.newaxis, :]
