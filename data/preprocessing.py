import os
import nrrd
import time
import torch
import argparse
import numpy as np
import scipy.ndimage
from glob import glob
import multiprocessing as mp
from torch.utils import data

import utils


def pp(scan):
    if os.path.splitext(scan)[1] == ".mhd":
        # Load scan and mask
        if args.verbose == True:
            print("Loading", scan)
        ct_scan, _, _, flip = utils.load_mhd(os.path.join(args.scans_path, scan))
        if flip:
            ct_scan = ct_scan[:, ::-1, ::-1]
            if args.verbose == True:
                print("->", scan, "flipped")
        seg_mask, _, _, flip = utils.load_mhd(os.path.join(args.labels_path, scan))
        if flip:
            seg_mask = seg_mask[:, ::-1, ::-1]
            if args.verbose == True:
                print("->", scan, "mask flipped")
        
        if args.n_classes == 3:
            seg_mask[seg_mask == 3] = 1
            seg_mask[seg_mask == 4] = 2
            seg_mask[seg_mask == 5] = 3
        if args.n_classes == 2:
            seg_mask[seg_mask == 3] = 1
            seg_mask[seg_mask == 4] = 2
            seg_mask[seg_mask > 2] = 0
            seg_mask[seg_mask < 1] = 0
        else:
            # Remove main bronchus
            seg_mask[seg_mask == 3] = 1
            seg_mask[seg_mask == 4] = 1
            seg_mask[seg_mask != 1] = 0

        ct_scan = scipy.ndimage.interpolation.zoom(ct_scan, [scan_size[0]/float(
            len(ct_scan)), scan_size[1]/512., scan_size[2]/512.], mode="nearest")
        seg_mask = scipy.ndimage.interpolation.zoom(seg_mask, [scan_size[0]/float(
            len(seg_mask)), scan_size[1]/512., scan_size[2]/512.], mode="nearest").astype(np.uint8)


        if args.n_classes == 3:
            seg_mask[seg_mask < 0] = 0
            seg_mask[seg_mask > 3] = 3
        if args.n_classes == 2:
            seg_mask[seg_mask < 0] = 0
            seg_mask[seg_mask > 2] = 2
        else:
            # Remove main bronchus
            seg_mask[seg_mask < 0] = 0
            seg_mask[seg_mask > 1] = 1
        
        # Writing scan and mask to npy
        np.save(os.path.join(args.output, scan.split('.')[0] + '.npy'), ct_scan)
        np.save(os.path.join(args.output, scan.split('.')[0] + '_mask.npy'), seg_mask)


if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser(description="Preprocessing script")
    parser.add_argument("-s", "--scans-path", required=True, help="Path to CT-scans directory.")
    parser.add_argument("-l", "--labels-path", required=True, help="Path to lungs segmentation labels directory.")
    parser.add_argument("-o", "--output", required=True, help="Output directory.")
    parser.add_argument("--n-classes", required=False, default=1, type=int, help="Number of classes.")
    parser.add_argument("-x", "--size-x", required=False, type=int, default=128, help="X size of preprocessed scan and label.")
    parser.add_argument("-y", "--size-y", required=False, type=int, default=256, help="Y size of preprocessed scan and label.")
    parser.add_argument("-z", "--size-z", required=False, type=int, default=256, help="Z size of preprocessed scan and label.")
    parser.add_argument("-v", "--verbose", action="store_true", required=False, help="If set, will output additional information.")
    args = parser.parse_args()

    start_time = time.time()
    
    # Create output directory if needed
    output_path = os.path.join(args.output, str(args.n_classes) + "_" + str(args.size_x) + "_" + str(args.size_y) + "_" + str(args.size_z))
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        args.output = output_path

        # Get scan_size from args
        scan_size = np.array([128, 256, 256])
        if args.size_x != None and args.size_y != None and args.size_z != None:
            scan_size = np.array([args.size_x, args.size_y, args.size_z])

        scans = os.listdir(args.scans_path)

        list_scans = np.unique([os.path.splitext(scans[i])[0] for i in range(len(scans))])
        np.save(os.path.join(args.output, "list_scans.npy"), list_scans)

        pool = mp.Pool(mp.cpu_count())
        pool.map(pp, scans)
        pool.close()

    print("--- Time (s): ", time.time() - start_time)

