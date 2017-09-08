# pf_geolocation
Particle filter geolocation in Python for Data Storage Tags

## Dependencies
This code requires the following packages:
* Basemap (pyproj)
* filterpy
* netcdf4
* pytides

For the GPU version the following are required in addition:
* CUDA (see [NVIDIA CUDA official documentation](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) for instructions)
* PyCUDA

## Preprocessing
Either of the following may be used to preprocess raw data from DSTs manufactured by Star-Oddi or Lotek:
* The preprocessing code for `hmm_smast`, available at https://github.com/cliu3/hmm_smast/tree/dev/test/preprocessing;
* An R code package; available at https://github.com/cliu3/R_HMM_Preprocessing. 
