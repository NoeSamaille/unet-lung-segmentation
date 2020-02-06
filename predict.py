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


def predict(scan_path, nb_classes, start_filters, output_path=None, img_out=None, threshold=False):

    # Use CUDA
    device = torch.device("cuda:0")

    # Load model
    unet = model.UNet(1, nb_classes, start_filters).to(device)
    unet.load_state_dict(torch.load("./model"))

    # Apply model to new scan
    _, scan_id = os.path.split(scan_path)
    scan_id = scan_id.split('.')[0]
    ct_scan, origin, spacing = utils.load_itk(scan_path)
    ct_scan = utils.prep_img_arr(ct_scan)
    x = torch.Tensor(np.array([ct_scan.astype(np.float16)])).to(device)
    logits = unet(x)
    mask = logits.cpu().detach().numpy()
    # Thresholding
    if threshold == True:
        # print(np.min(mask))
        # print(np.max(mask))
        # print(np.mean(mask))
        # print(np.std(mask))
        ts = np.mean(mask)+np.std(mask)
        mask[mask <= ts] = 0
        mask[mask > ts] = 1
    if output_path is not None:
        utils.write_itk(os.path.join(output_path, scan_id + '_mask.nrrd'), mask[0][0], origin, spacing)
        if img_out is not None:
            utils.write_itk(os.path.join(output_path, scan_id + '.nrrd'), ct_scan[0], origin, spacing)
    return mask

if __name__ == "__main__":
    
    # Argument parser
    parser = argparse.ArgumentParser(description="Inference script")
    parser.add_argument("-d", "--data", required=True, help="Path to input CT-scan (nrrd format)")
    parser.add_argument("-o", "--output", required=True, help="Path to output directory")
    parser.add_argument("-c", "--nb-classes", required=True, help="Number of U-Net output classes")
    parser.add_argument("-f", "--start-filters", required=True, help="Number of filters")
    parser.add_argument("--image-out", action="store_true", required=False, help="If set, ct-scan will be written in output directory")
    parser.add_argument("-t", "--threshold", action="store_true", required=False, help="If set, resulting mask will be thresholded by mean+1sigma")
    args = parser.parse_args()

    predict(args.data, int(args.nb_classes), int(args.start_filters), output_path=args.output, img_out=args.image_out, threshold=args.threshold)

