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

__device__ int unique(int *array, int Length) {
    // find unique elements of an array, ignore zeros
    // code midified from https://stackoverflow.com/a/1533394/4695588
  int i, j;
  // int array[12] = {1, 2, 4, 4, 3, 6, 5, 2, 5, 3, 9, 10};
  // int Length = 12;
  /* new length of modified array */
  int NewLength = 1;

  for (i = 1; i < Length; i++) {
    for (j = 0; j < NewLength; j++) {
      if (array[i] == array[j] || array[i] == 0) break;
    }

    /* if none of the values in index[0..j] of array is not same as array[i],
       then copy the current value to corresponding new position in array */

    if (j == NewLength) array[NewLength++] = array[i];
  }
  return NewLength;
}

__device__ void merge(int *a, int nbr_a, int *b, int nbr_b, int *c) {
    // merge two arrays
    // code midified from https://stackoverflow.com/a/1700335/4695588
    int i=0, j=0, k=0;

    // Phase 1) 2 input arrays not exhausted
    while( i<nbr_a && j<nbr_b )
    {
        if( a[i] <= b[j] )
            c[k++] = a[i++];
        else
            c[k++] = b[j++];
    }

    // Phase 2) 1 input array not exhausted
    while( i < nbr_a )
        c[k++] = a[i++];
    while( j < nbr_b )
        c[k++] = b[j++];

}

__global__ void update_loc(float * x, float * y, const float *x0, const float *y0, const float *xv, const float *yv, const unsigned int *nv, const unsigned int *ntve, const unsigned int *nbve, const int *minloc, const int *second_minloc, const int nele, const int N) 
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int i, j;
    int NewLength;
    int stat = 1;
    float xtri[3];
    float ytri[3];
    int nbve1[9];
    int nbve2[9];
    int ntve1 = ntve[minloc[idx]];
    int ntve2 = ntve[second_minloc[idx]];
    int nbve_total[20];
    
    if(idx >= N)
        return;

    //collece nbve's
    for (i=0; i<ntve1; i++) 
        nbve1[i] = nbve[i * nele + minloc[idx]];
    for (i=0; i<ntve2; i++) 
        nbve2[i] = nbve[i * nele + second_minloc[idx]];

    //merge and unique
    merge(nbve1, ntve1, nbve2, ntve2, nbve_total);
    NewLength = unique(nbve_total, ntve1+ntve2);

    // find if particle is in any of the cells surrounding the two nearest nodes
    for (j=0; j<NewLength; j++) {

        for (i=0; i<3; i++) {
            xtri[i] = xv[nv[i * nele + nbve_total[j]]];
            ytri[i] = yv[nv[i * nele + nbve_total[j]]];
            
            if (isintriangle(xtri,ytri,x[idx],y[idx])) {
                stat = 1;
                break;
            } else { 
                stat = 0;
            }

        }

        if (stat == 0) {
            x[idx] = x0[idx];
            y[idx] = y0[idx];        
        }

    }


}
