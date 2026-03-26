
// Functions Headers
#include "Vector_Addition.cuh"
// CUDA Libary
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
// C++ Headers
#include <chrono>
#include <iostream>
#include <stdio.h>

int main()
{
    int N = 1 << 20; // Size of vectors ( 1 million elements)
    size_t size = N * sizeof(float);

    // Allocate host memory
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    // Initialize input vectors
    for (int i = 0; i < N; i++)
    {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(2*i);
    }

    // Allocate device memory
    float* d_A, *d_B, *d_C = nullptr;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy input data from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Lanuch kernel function on GPU
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Synchronize threads
    cudaDeviceSynchronize();

    // Copy result from device to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Verification
    bool sucess = true;
    for (int i = 0; i < N; i++)
    {
        float expected = h_A[i] + h_B[i];
        if (fabs(h_C[i] - expected) > 1e-5)
        {
            sucess = false;
            break;
        }
    }
    if (sucess)
    {
        std::cout << "Vector addition completed sucessfully." << std::endl;
    }
    else
    {
        std::cout << "Error in vector addition." << std::endl;
    }

    // Free Device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free Host memory
    free(h_A);
    free(h_B); 
    free(h_C);

    return 0;
}
