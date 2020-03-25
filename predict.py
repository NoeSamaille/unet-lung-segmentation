import numpy as np
import pickle, nrrd, json
import SimpleITK as sitk
import torch
from torch.utils import data
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F
from scipy import ndimage
import argparse
import glob
import os

try:
    import model
    from data import *
except:
    from . import model
    from .data import *


def predict(ct_scan, nb_classes, start_filters, model_path, threshold=False, 
            erosion=False, verbose=False):

    # Use CUDA
    device = torch.device("cuda:0")
    # LMS
    try:
        torch.cuda.set_enabled_lms(True)
    except:
        print("LMS not supported")
    # Load model
    unet = model.UNet(1, nb_classes, start_filters).to(device)
    unet.load_state_dict(torch.load(model_path))

    # Apply model to new scan
    x = torch.Tensor(np.array([ct_scan.astype(np.float16)])).to(device)
    logits = unet(x)
    mask = logits.cpu().detach().numpy()

    # Thresholding
    if threshold == True:
        if nb_classes > 1:
            mask = np.round(mask).astype('uint8')
        else:
            ts = np.mean(mask) + np.std(mask)
            mask[mask <= ts] = 0
            mask[mask > ts] = 1
            mask = mask.astype('uint8')

    # Morphological closing (dilation -> erosion)
    mask[0][0] = ndimage.binary_closing(mask[0][0], 
        structure=ndimage.generate_binary_structure(3,2)).astype(mask.dtype)

    # Morphological erosion
    if erosion == True:
        mask[0][0] = ndimage.binary_erosion(mask[0][0], 
            structure=ndimage.generate_binary_structure(3,2)).astype(mask.dtype)

    return mask


if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser(description="Inference script")
    parser.add_argument("-d", "--data", required=True, help="Path to input CT-scan (nrrd format)")
    parser.add_argument("-o", "--output", required=True, help="Path to output directory")
    parser.add_argument("-c", "--nb-classes", required=True, help="Number of U-Net output classes")
    parser.add_argument("-f", "--start-filters", required=True, help="Number of filters")
    parser.add_argument("-m", "--model", required=True, help="Path to model")
    parser.add_argument("-t", "--threshold", action="store_true", required=False, help="If set, resulting mask will be thresholded by mean+sigma")
    parser.add_argument("-e", "--erosion", action="store_true", required=False, help="If set, will perform morphological erosion on resulting mask")
    parser.add_argument("-x", "--scan-size-x", type=int, default=128, required=False, help="X size of resized CT-scan")
    parser.add_argument("-y", "--scan-size-y", type=int, default=256, required=False, help="Y size of resized CT-scan")
    parser.add_argument("-z", "--scan-size-z", type=int, default=256, required=False, help="Z size of resized CT-scan")
    parser.add_argument("-v", "--verbose", action="store_true", required=False, help="If set, will show additionnal information")
    args = parser.parse_args()

    # Load scan
    _, scan_id = os.path.split(args.data)
    scan_id = scan_id.split('.')[0]
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
    mask = predict(ct_scan, int(args.nb_classes), int(args.start_filters), args.model, threshold=args.threshold, erosion=args.erosion, verbose=args.verbose)
    
    if int(args.nb_classes) > 1:
        mask = mask[0][1]
    else:
        mask = mask[0][0]

    # Resample mask
    mask = utils.resample(mask, spacing, orig_spacing).astype('uint8')

    if args.threshold == True:
        if int(args.nb_classes) > 1:
            mask[mask<0] = 0
            mask[mask>2] = 2
        else:
            mask[mask<=0] = 0
            mask[mask>0] = 1

    if args.verbose == True:
        print("Mask:\n -> shape:", mask.shape, "\n -> spacing:", orig_spacing, "\n -> unique:", np.unique(mask))

    # Write into ouput files (nrrd format)
    utils.write_itk(os.path.join(args.output, scan_id + '_mask.nrrd'), mask, origin, orig_spacing)
