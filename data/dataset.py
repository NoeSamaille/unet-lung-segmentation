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
        - list_scans: list containing the filenames of scans
        - scans_path: path to the scans
        - masks_path: path to the masks corresponding to the scans
        - mode: 2d will return slices
        - scan_size: size of CT-scans
        - n_classes: number of output class
    """

    def __init__(self, list_scans, scans_path, masks_path, mode="3d", scan_size=[128, 256, 256], n_classes=1):
        self.list_scans = list_scans
        self.scans_path = scans_path
        self.masks_path = masks_path
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
        path = os.path.join(self.scans_path, scan, '*', '*')
        # used to find the corresponding lung mask
        scan_dicom_id = os.path.basename(glob(path)[0])
        if verbose == True:
            print("DICOM id:", scan_dicom_id)
        # tuple containing the CT scan and some metadata
        nrrd_scan = nrrd.read(glob(os.path.join(path, "*CT.nrrd"))[0])
        ct_scan = np.swapaxes(nrrd_scan[0], 0, 2)
        # function uses SimpleITK to load lung masks from mhd/zraw data
        seg_mask, _, _ = utils.load_itk(os.path.join(
            self.masks_path, scan_dicom_id + ".mhd"))

        if self.n_classes == 3:
            seg_mask[seg_mask == 3] = 1
            seg_mask[seg_mask == 4] = 2
            seg_mask[seg_mask == 5] = 3
        else:
            seg_mask[seg_mask <= 0] = 0
            seg_mask[seg_mask > 0] = 1

        if verbose == True:
            nrrd.write("./luna16_mask.nrrd", np.swapaxes(seg_mask, 0, 2))

        if self.mode == "3d":
            ct_scan = scipy.ndimage.interpolation.zoom(ct_scan, [self.scan_size[0]/float(
                len(ct_scan)), self.scan_size[1]/512., self.scan_size[2]/512.], mode="nearest")
            seg_mask = scipy.ndimage.interpolation.zoom(seg_mask, [self.scan_size[0]/float(
                len(seg_mask)), self.scan_size[1]/512., self.scan_size[2]/512.], mode="nearest")

        if self.mode == "2d":
            return ct_scan[:, np.newaxis, :], seg_mask[:, np.newaxis, :]
        else:
            return ct_scan[np.newaxis, :], seg_mask[np.newaxis, :]
