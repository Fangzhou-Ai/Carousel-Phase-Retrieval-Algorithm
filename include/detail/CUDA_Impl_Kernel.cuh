#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#include <complex>
namespace CPRA
{
namespace Kernel
{
// Kernels can be optimized
// These are demos
template<typename T>
__global__ void ker_SpaceConstraint(thrust::complex<T>* flat_src_data, T* flat_constr_data, size_t num, size_t batch_size)
{
    int thread_id =  blockIdx.x * blockDim.x + threadIdx.x;
    for(size_t i = 0; i < num; i+= gridDim.x * blockDim.x)
    {
        flat_src_data[i].imag(0);
        if(flat_constr_data[i / batch_size] == 0)
            flat_src_data[i].real(0);
    }
}



}
}