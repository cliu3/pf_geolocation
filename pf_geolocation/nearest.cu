// brute force nearest neighbor
//original code from: http://nghiaho.com/?p=416
#include <float.h>
#define CUDA_NN_DIM 2 // data dimension

__global__ void nearest(const float *query_x, const float *query_y, int query_pts, const float *data, int data_pts,
                    int *idxs)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    float dist_sq;

    if(idx >= query_pts)
        return;

    int best_idx = -1;
    float best_dist = FLT_MAX;

    for(int i=0; i < data_pts; i++) {
        dist_sq = 0;

        
        float d = query_x[idx] - data[i*CUDA_NN_DIM];
        dist_sq += d*d;
        d = query_y[idx] - data[i*CUDA_NN_DIM+1];
        dist_sq += d*d;
        

        if(dist_sq < best_dist) {
            best_dist = dist_sq;
            best_idx = i;
        }
    }

    idxs[idx] = best_idx;
//    dist_sq[idx] = best_dist;
}
