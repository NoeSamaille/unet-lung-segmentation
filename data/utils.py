import numpy as np
import torch
import SimpleITK as sitk
import scipy
import skimage


# Not sure if works for all format (Tested only on mhd/zraw and nrrd format)
def load_itk(filename):
    itkimage = sitk.ReadImage(filename)
    ct_scan = sitk.GetArrayFromImage(itkimage).astype(np.int16)
    origin = np.array(itkimage.GetOrigin())
    spacing = np.array(itkimage.GetSpacing())
    return ct_scan, origin, spacing


# Tested on nrrd format
def write_itk(output_path, img_array, origin, spacing):
    itk_image = sitk.GetImageFromArray(img_array)
    itk_image.SetOrigin(origin)
    itk_image.SetSpacing(spacing)
    sitk.WriteImage(itk_image, output_path)


# Resample img to match target_spacing
def resample(img, spacing, target_spacing):
    target_shape = [int(img.shape[i] * spacing[i] / target_spacing[i]) for i in range(len(img.shape))]
    img = skimage.transform.resize(img, target_shape, order=1, clip=True, mode='edge')
    return img


# Prepare image for model
def prep_img_arr(img, spacing, target_shape=[128, 256, 256]):
    target_spacing = [img.shape[i] * spacing[i] / target_shape[i] for i in range(len(img.shape))]
    #img = scipy.ndimage.interpolation.zoom(img, zoom_factors, mode="nearest")
    img = resample(img, spacing, target_spacing)
    return img[np.newaxis, :], target_spacing


def dice_loss(logits, labels, eps=1e-7):
    '''
      logits, labels, shape : [B, 1, Y, X]

    '''
    num = 2. * torch.sum(logits * labels)
    denom = torch.sum(logits**2 + labels**2)
    return 1 - torch.mean(num / (denom + eps))
