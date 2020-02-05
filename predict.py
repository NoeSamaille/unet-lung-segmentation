import numpy as np
import pickle, nrrd, json
import SimpleITK as sitk
import torch
from torch.utils import data
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F
import argparse

import model
from data import *


def predict(scan_path, output_path, nb_classes, start_filters, to_nrrd=False, img_out=None):

    # Use CUDA
    device = torch.device("cuda:0")
    
    # Load model
    unet = model.UNet(1, nb_classes, start_filters).to(device)
    unet.load_state_dict(torch.load("./model"))
    
    # Apply model to new scan
    ct_scan, origin, spacing = utils.load_itk(scan_path)
    ct_scan = utils.prep_img_arr(ct_scan)
    x = torch.Tensor(np.array([ct_scan.astype(np.float16)])).to(device)
    logits = unet(x)
    mask = logits.cpu().detach().numpy()
    utils.write_itk(output_path, mask[0][0], origin, spacing)
    if img_out is not None:
        utils.write_itk(img_out, ct_scan[0], origin, spacing) 


if __name__ == "__main__":
    
    # Argument parser
    parser = argparse.ArgumentParser(description="Inference script")
    parser.add_argument("-d", "--data", required=True, help="Path to input CT-scan (nrrd format)")
    parser.add_argument("-o", "--output", required=True, help="Path to output directory")
    parser.add_argument("-c", "--nb-classes", required=True, help="")
    parser.add_argument("-f", "--start-filters", required=True, help="")
    parser.add_argument("--io", required=False, help="")
    args = parser.parse_args()

    predict(args.data, args.output, int(args.nb_classes), int(args.start_filters), img_out=args.io)

