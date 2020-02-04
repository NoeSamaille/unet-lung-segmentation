# Lung Segmentation

Lung Segmentation using a UNet model on 3D CT scans.

## Current results example :

![lung segmentation example](https://github.com/Thvnvtos/Lung_Segmentation/blob/unet3d/readme_images/example_segmentation.png?raw=true)

## Getting started

### Installation

Our base wmlce conda environment does not come with `SimpleITK` nor `pynrrd`, two required python libraries to run this code.

+ To install `pynrrd`:
```
$ pip install pynrrd
```

+ To install `SimpleITK` (need to build the library on Power):
```
$ conda update conda
$ conda update conda-build
$ git clone https://github.com/SimpleITK/SimpleITKCondaRecipe.git
$ cd SimpleITKCondaRecipe
$ conda build recipe
$ conda install -c file://PATH_TO_ANACONDA/conda-bld simpleitk
```
  + `PATH_TO_ANACONDA` example on powerai wmlce: `/opt/anaconda/envs/wmlce/conda-bld`

### Tree

```
.
+-- data/
    +-- dataset.py	: Class describing the dataset we use for lung segmentation
    +-- utils.py	: Script for manipulating medical files
+-- config.json
+-- eval.py
+-- model.py		: U-Net model definition
+-- predict.py		: Inference script to run infer lung mask on a CT-scan
+-- README.md		: This documentation file
+-- train.py		: Train script to train a new lung segmentation model
```

### Data 

The data used is the __TCIA LIDC-IDRI__ dataset Standardized representation ([download here](https://wiki.cancerimagingarchive.net/display/DOI/Standardized+representation+of+the+TCIA+LIDC-IDRI+annotations+using+DICOM)), combined with matching lung masks from __LUNA16__ (not all CT-scans have their lung masks in LUNA16 so we need the list of segmented ones).

3 parameters have to be fulfilled to use available data:
+ `labelled-list`: path to the `pickle` file containing the list of CT-scans from the TCIA LIDC-IDRI dataset for which we have access to the lung segmentation masks through the LUNA16 dataset.
+ `scans`: path to the TCIA LIDC-IDRI dataset.
+ `masks`: path to the LUNA16 dataset containing lung masks.

You can manipulate data trough the `data/dataset.py` (class describing our lung segmentation dataset) and `data/utils.py` (tools for manipulating medical files) files.

### Run predictions

To perform predictions using the existing model run for example (wmlce on powerai):
```
$ export LABELLED_LIST=/wmlce/data/retina-unet/data/labelled.pickle
$ export MASKS=/wmlce/data/retina-unet/data/lung_masks_LUNA16
$ export SCANS=/wmlce/data/retina-unet/data/LIDC-IDRI
$ export NB_CLASSES=1
$ export START_FILTERS=32
$ pyhton3 predict.py --labelled-list $LABELLED_LIST --masks $MASKS --scans $SCANS --nb-classes $NB_CLASSES --start-filters $START_FILTERS 
``` 
