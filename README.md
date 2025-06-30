# UDO

Code for paper "Optimising Urban Satellite Disparity Estimation via Unsupervised Local Geometric Constraints".

# Environment

* cv2
* GDAL
* h5py
* numpy
* tqdm
* subprocess
* matplotlib

# Usage

#### Preparing for superpixel segmentation

![](Superpixel.png)

```
python Superpixel/main.py --left_img <left>.tiff --disp_img <disp>.tiff --name <name>
```

#### Optimising

```
python Train.py
```

