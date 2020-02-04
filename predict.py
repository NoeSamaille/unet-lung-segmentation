import numpy as np
import pickle, nrrd, json
import torch
from torch.utils import data
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F
import argparse

import model
from data import *


def predict(labelled_list, masks, scans, nb_classes, start_filters):

    # Use CUDA
    device = torch.device("cuda:0")
    
    # Open labelled list
    with open(labelled_list, "rb") as f:
        list_scans = pickle.load(f)
    
    # Parse list of CT-scans
    ct_scans = [s.split('/')[1] for s in list_scans]
    ct_scans = ct_scans[30:]

    # Build dataset
    data = dataset.Dataset(ct_scans, scans, masks, mode="3d")
    
    # Load model
    criterion = utils.dice_loss
    unet = model.UNet(1, nb_classes, start_filters).to(device)
    unet.load_state_dict(torch.load("./model"))
    
    # Apply model to new scan
    x,y = data.__getitem__(15, verbose=True)
    x = torch.Tensor(np.array([x.astype(np.float16)])).to(device)
    y = torch.Tensor(np.array([y.astype(np.float16)])).to(device)
    logits = unet(x)
    loss = criterion(logits, y)
    print(loss.item())
    mask = logits.cpu().detach().numpy()
    nrrd.write("lung_mask2.nrrd", mask[0][0])


if __name__ == "__main__":
    
    # Argument parser
    parser = argparse.ArgumentParser(description="Inference script")
    parser.add_argument("-l", "--labelled-list", required=True, help="List of labelled lungs")
    parser.add_argument("-m", "--masks", required=True, help="")
    parser.add_argument("-s", "--scans", required=True, help="")
    parser.add_argument("-c", "--nb-classes", required=True, help="")
    parser.add_argument("-f", "--start-filters", required=True, help="")
    args = parser.parse_args()

    predict(args.labelled_list, args.masks, args.scans, int(args.nb_classes), int(args.start_filters))

