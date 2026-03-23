#include "cuda_runtime.h"

#include <iostream>
#include <stdio.h>


__global__ void vextorAdd(const float* A, const float* B, float* C, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < N)
    {
        C[i] = A[i] + B[i];
    }
}
