import numpy as np
import torch
import SimpleITK as sitk
import scipy


# Not sure if works for all format (Tested only on mhd/zraw and nrrd format)
def load_itk(filename):
    itkimage = sitk.ReadImage(filename)
    ct_scan = sitk.GetArrayFromImage(itkimage)
    origin = np.array(list(reversed(itkimage.GetOrigin())))
    spacing = np.array(list(reversed(itkimage.GetSpacing())))
    return ct_scan, origin, spacing


# Tested on nrrd format
def write_itk(output_path, img_array, origin, spacing):
    itk_image = sitk.GetImageFromArray(img_array)
    itk_image.SetOrigin(origin)
    itk_image.SetSpacing(spacing)
    sitk.WriteImage(itk_image, output_path)


def prep_img_arr(img_array, scan_size=[128, 256, 256]):
    img_array = scipy.ndimage.interpolation.zoom(img_array, [scan_size[0]/float(len(img_array)), scan_size[1]/512., scan_size[2]/512.], mode="nearest")
    return img_array[np.newaxis, :]


def dice_loss(logits, labels, eps=1e-7):
    '''
      logits, labels, shape : [B, 1, Y, X]

    '''
    num = 2. * torch.sum(logits * labels)
    denom = torch.sum(logits**2 + labels**2)
    return 1 - torch.mean(num / (denom + eps))
