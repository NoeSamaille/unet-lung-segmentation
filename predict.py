import numpy as np
import pickle, nrrd, json
import SimpleITK as sitk
import torch
from torch.utils import data
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F
import argparse
import glob
import os

try:
    import model
    from data import *
except:
    from . import model
    from .data import *


def predict(ct_scan, nb_classes, start_filters, model_path, threshold=False, verbose=False):

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
        ts = np.mean(mask) + np.std(mask)
        mask[mask <= ts] = 0
        mask[mask > ts] = 1
        mask = mask.astype('uint8')
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
    parser.add_argument("-v", "--verbose", action="store_true", required=False, help="If set, will show additionnal information")
    args = parser.parse_args()

    # Load scan
    _, scan_id = os.path.split(args.data)
    scan_id = scan_id.split('.')[0]
    ct_scan, origin, orig_spacing = utils.load_itk(args.data)
    if args.verbose == True:
        print(scan_id, ":\n -> shape:", ct_scan.shape, "\n -> spacing:", orig_spacing)
    ct_scan, spacing = utils.prep_img_arr(ct_scan, orig_spacing)
    if args.verbose == True:
        print("CT-scan:\n -> shape:", ct_scan.shape, "\n -> spacing:", spacing)

    # Compute lungs mask
    mask = predict(ct_scan, int(args.nb_classes), int(args.start_filters), args.model, threshold=args.threshold, verbose=args.verbose)

    # Resample mask
    mask = utils.resample(mask[0][0], spacing, orig_spacing)
    if args.threshold == True:
        mask[mask<=0] = 0
        mask[mask>0] = 1
        mask = mask.astype('uint8')
    if args.verbose == True:
        print("Mask:\n -> shape:", mask.shape, "\n -> spacing:", orig_spacing)
    # Write into ouput files (nrrd format)
    utils.write_itk(os.path.join(args.output, scan_id + '_mask.nrrd'), mask, origin, orig_spacing)
