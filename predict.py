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

import model
from data import *


def predict(ct_scan, nb_classes, start_filters, threshold=False, verbose=False):

    # Use CUDA
    device = torch.device("cuda:0")

    # Load model
    unet = model.UNet(1, nb_classes, start_filters).to(device)
    unet.load_state_dict(torch.load("./model"))

    # Apply model to new scan
    x = torch.Tensor(np.array([ct_scan.astype(np.float16)])).to(device)
    logits = unet(x)
    mask = logits.cpu().detach().numpy()
    if verbose == True:
        print(np.shape(mask), np.min(mask), np.max(mask), np.mean(mask), np.std(mask))

    # Thresholding
    if threshold == True:
        ts = np.mean(mask)+np.std(mask)
        mask[mask <= ts] = 0
        mask[mask > ts] = 1
    return mask


if __name__ == "__main__":
    
    # Argument parser
    parser = argparse.ArgumentParser(description="Inference script")
    parser.add_argument("-d", "--data", required=True, help="Path to input CT-scan (nrrd format)")
    parser.add_argument("-o", "--output", required=True, help="Path to output directory")
    parser.add_argument("-c", "--nb-classes", required=True, help="Number of U-Net output classes")
    parser.add_argument("-f", "--start-filters", required=True, help="Number of filters")
    parser.add_argument("-t", "--threshold", action="store_true", required=False, help="If set, resulting mask will be thresholded by mean+sigma")
    parser.add_argument("-v", "--verbose", action="store_true", required=False, help="If set, will show additionnal information")
    args = parser.parse_args()

    # Load scan
    _, scan_id = os.path.split(args.data)
    scan_id = scan_id.split('.')[0]
    ct_scan, origin, spacing = utils.load_itk(args.data)
    ct_scan = utils.prep_img_arr(ct_scan)
    if args.verbose == True:
        print(np.shape(ct_scan))

    # Compute lungs mask
    mask = predict(ct_scan, int(args.nb_classes), int(args.start_filters), threshold=args.threshold, verbose=args.verbose)

    # Write into ouput files (nrrd format)
    if args.output is not None:
        utils.write_itk(os.path.join(args.output, scan_id + '_mask.nrrd'), mask[0][0], origin, spacing)
        utils.write_itk(os.path.join(args.output, scan_id + '.nrrd'), ct_scan[0], origin, spacing)
