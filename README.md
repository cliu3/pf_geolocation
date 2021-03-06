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

# Publications
Liu, C., Cowles, G.W., Zemeckis, D.R., Fay, G., Le Bris, A., Cadrin, S.X. (2019), A hardware-accelerated particle filter for the geolocation of demersal fish. Fisheries Research, 213C(2019):160-171. [doi:10.1016/j.fishres.2019.01.019](https://dx.doi.org/10.1016/j.fishres.2019.01.019).
