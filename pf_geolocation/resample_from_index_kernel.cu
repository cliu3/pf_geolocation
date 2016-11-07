__global__ void resample_from_index_kernel(float * x, float * y, const float *x0, const float *y0, const unsigned int *indexes, const int N) 
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    //float x_kernel, y_kernel;

    if(idx >= N)
        return;

    //x_kernel = x0[indexes[idx]];
    //y_kernel = y0[indexes[idx]];

    //__syncthreads();

    //x[idx] = x_kernel;
    //y[idx] = y_kernel;
    
    x[idx] = x0[indexes[idx]];
    y[idx] = y0[indexes[idx]];
}
