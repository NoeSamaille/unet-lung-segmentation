#!/usr/bin/env python
import numpy as np
import pickle, nrrd, json
import SimpleITK as sitk
import torch
import torch.nn as nn
from torch.utils import data
import torch.optim as optim
from torch.nn import functional as F
from scipy import ndimage
import mlflow
import mlflow.pytorch
import argparse
import glob
import os


def predict(ct_scan, nb_classes, unet, threshold=False, 
            erosion=False, verbose=False):

    # Use CUDA
    device = torch.device("cuda:0")
    # LMS
    try:
        torch.cuda.set_enabled_lms(True)
    except:
        print("LMS not supported")
    # Load model
    unet = unet.to(device)

    # Apply model to new scan
    x = torch.Tensor(np.array([ct_scan.astype(np.float16)])).to(device)
    logits = unet(x)
    mask = logits.cpu().detach().numpy()

    # Thresholding
    if threshold == True:
        struct = ndimage.generate_binary_structure(3,2)
        if nb_classes == 2:
            mask = np.round(mask).astype('uint8')
            
            # # Separate two lungs
            # two_lungs_mask = mask[0][1]
            # lung1_mask = np.zeros(two_lungs_mask.shape)
            # lung1_mask[two_lungs_mask == 1] = 1
            # lung2_mask = np.zeros(two_lungs_mask.shape)
            # lung2_mask[two_lungs_mask == 2] = 1

            # lung1_mask = ndimage.binary_opening(lung1_mask, structure=struct, iterations=5)
            
            # lung2_mask = ndimage.binary_opening(lung2_mask, structure=struct, iterations=5)
            
            # mask[0][1].fill(0)
            # mask[0][1][lung1_mask == 1] = 1
            # mask[0][1][lung2_mask == 1] = 2
        else:
            ts = np.mean(mask) + np.std(mask)
            mask[mask <= ts] = 0
            mask[mask > ts] = 1
            mask = mask.astype('uint8')

            # Morphological closing (dilation -> erosion)
            mask[0][0] = ndimage.binary_closing(mask[0][0], structure=struct).astype(mask.dtype)

            # Morphological erosion
            if erosion == True:
                mask[0][0] = ndimage.binary_erosion(mask[0][0], structure=struct).astype(mask.dtype)

    return mask


if __name__ == "__main__":

    try:
        import model
        from data import *
    except:
        from . import model
        from .data import *

    # Argument parser
    parser = argparse.ArgumentParser(description="Inference script")
    parser.add_argument("-d", "--data", required=True, help="Path to input CT-scan (nrrd format)")
    parser.add_argument("-o", "--output", required=True, help="Path to output directory")
    parser.add_argument("-c", "--nb-classes", required=False, type=int, default=1, help="Number of U-Net output classes")
    parser.add_argument("-f", "--start-filters", required=False, type=int, default=32, help="Number of filters")
    parser.add_argument("-m", "--model", required=False, help="Path to model")
    parser.add_argument("-t", "--threshold", action="store_true", required=False, help="If set, resulting mask will be thresholded by mean+sigma")
    parser.add_argument("-e", "--erosion", action="store_true", required=False, help="If set, will perform morphological erosion on resulting mask")
    parser.add_argument("-x", "--scan-size-x", type=int, default=128, required=False, help="X size of resized CT-scan")
    parser.add_argument("-y", "--scan-size-y", type=int, default=256, required=False, help="Y size of resized CT-scan")
    parser.add_argument("-z", "--scan-size-z", type=int, default=256, required=False, help="Z size of resized CT-scan")
    parser.add_argument("-v", "--verbose", action="store_true", required=False, help="If set, will show additionnal information")
    args = parser.parse_args()
    
    # MLFlow setup
    if not args.model:
        remote_server_uri = "http://mlflow.10.7.13.202.nip.io/"
        mlflow.set_tracking_uri(remote_server_uri)

    # Load scan
    scan_id = os.path.basename(args.data).split('.')[0]
    ct_scan, origin, orig_spacing = utils.load_itk(args.data)
    if args.verbose == True:
        print(scan_id, ":\n -> shape:", ct_scan.shape, "\n -> spacing:", orig_spacing)
    target_shape = np.array([128, 256, 256])
    if args.scan_size_x != None and args.scan_size_y != None and args.scan_size_z != None:
        target_shape = np.array([args.scan_size_x, args.scan_size_y, args.scan_size_z])
    ct_scan, spacing = utils.prep_img_arr(ct_scan, orig_spacing, target_shape)
    if args.verbose == True:
        print("CT-scan:\n -> shape:", ct_scan.shape, "\n -> spacing:", spacing)

    # Compute lungs mask
    if args.model:
        unet = model.UNet(1, args.nb_classes, args.start_filters)
        unet.load_state_dict(torch.load(args.model))
    else:
        if args.nb_classes == 2:
            model_name = "lung-segmentation"
        else:
            model_name = "2-lungs-segmentation"
        unet = mlflow.pytorch.load_model("models:/{}/production".format(model_name))
    mask = predict(ct_scan, args.nb_classes, unet, threshold=args.threshold, erosion=args.erosion, verbose=args.verbose)
    
    if args.nb_classes > 1:
        mask = mask[0][1]
    else:
        mask = mask[0][0]

    # Resample mask
    mask = utils.resample(mask, spacing, orig_spacing).astype('uint8')

    if args.threshold == True:
        if args.nb_classes == 2:
            mask[mask<0] = 0
            mask[mask>2] = 2
        else:
            mask[mask<=0] = 0
            mask[mask>0] = 1

    if args.verbose == True:
        print("Mask:\n -> shape:", mask.shape, "\n -> spacing:", orig_spacing, "\n -> unique:", np.unique(mask))

    # Write into ouput files (nrrd format)
    utils.write_itk(os.path.join(args.output, scan_id + '_mask.nrrd'), mask, origin, orig_spacing)
