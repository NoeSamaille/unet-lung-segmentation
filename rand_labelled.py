from glob import glob
from data import *
import numpy as np
import pickle
import nrrd
import os

if __name__ == "__main__":
    labelled_list = "/wmlce/data/retina-unet/data/labelled.pickle"
    scans_path = "/wmlce/data/retina-unet/data/LIDC-IDRI"
    masks_path = "/wmlce/data/retina-unet/data/lung_masks_LUNA16"
    with open(labelled_list, "rb") as f:
        list_scans = pickle.load(f)
    ct_scans = [s.split('/')[1] for s in list_scans]
    scan = ct_scans[np.random.choice(len(list_scans), 1)[0]]
    print(scan)
    scan_dicom_id = os.path.basename(glob(os.path.join(scans_path, scan, '*', '*'))[0])
    print(scan_dicom_id)
    seg_mask, origin, spacing = utils.load_itk(os.path.join(masks_path, scan_dicom_id + ".mhd"))
    seg_mask[seg_mask <= 0] = 0
    seg_mask[seg_mask > 0] = 1
    nrrd.write("./luna16_mask.nrrd", np.swapaxes(seg_mask, 0, 2))
