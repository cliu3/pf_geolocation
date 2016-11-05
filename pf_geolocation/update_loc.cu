__device__ int isintriangle(float xt[3], float yt[3], float x0, float y0)
//__device__ int isintriangle(const float *xt[3], const float *yt[3], const float *x0, const float *y0) 
{
    int res = 0;
    float f1, f2, f3;
    
    f1 = (y0-yt[0])*(xt[1]-xt[0]) - (x0-xt[0])*(yt[1]-yt[0]);
    f2 = (y0-yt[2])*(xt[0]-xt[2]) - (x0-xt[2])*(yt[0]-yt[2]);
    f3 = (y0-yt[1])*(xt[2]-xt[1]) - (x0-xt[1])*(yt[2]-yt[1]);
    if(f1*f3 >= 0.0 && f3*f2 >= 0.0) {
        res = 1;
    }
    return res;

}

__global__ void update_loc(float * x, float * y, const float *x0, const float *y0, const float *xv, const float *yv, const int *nv, const int *minloc, const int nnodes, const int N) 
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int i;
    float xtri[3];
    float ytri[3];
    
    if(idx >= N)
        return;

    for (i=0; i<3; i++) {
        xtri[i] = xv[nv[i * nnodes + minloc[idx]]];
        ytri[i] = yv[nv[i * nnodes + minloc[idx]]];
    }

    if (!isintriangle(xtri,ytri,x[idx],y[idx])) {
        x[idx] = x0[idx];
        y[idx] = y0[idx];
    }


}
