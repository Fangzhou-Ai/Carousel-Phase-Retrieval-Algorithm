#pragma once
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cuda.h>
#include <complex>
namespace cg = cooperative_groups;

namespace CPRA
{
namespace Kernel
{
// Kernels can be optimized
// These are demos
template<typename T>
__global__ void ker_SpaceConstraint(thrust::complex<T>* flat_src_data, T* flat_constr_data, uint64_t num, uint64_t batch_size)
{
    
    for(uint64_t i = cg::this_grid().thread_rank(); i < num; i+= cg::this_grid().size())
    {
        flat_src_data[i].imag(0);
        if(flat_constr_data[i % (num / batch_size)] == 0)
            flat_src_data[i].real(0);
    }
}

template<typename T>
__global__ void ker_RealDataConstraint(thrust::complex<T>* flat_src_data, T* flat_constr_data, uint64_t num, uint64_t batch_size)
{
    for(uint64_t i = cg::this_grid().thread_rank(); i < num; i+= cg::this_grid().size())
    {
        flat_src_data[i] = thrust::polar<T, T>(flat_constr_data[i % (num / batch_size)], thrust::arg(flat_src_data[i]));
    }
}

template<typename T>
__global__ void ker_ComplexDataConstraint(thrust::complex<T>* flat_src_data, thrust::complex<T>* flat_constr_data, uint64_t num, uint64_t batch_size)
{
    for(uint64_t i = cg::this_grid().thread_rank(); i < num; i+= cg::this_grid().size())
    {
        if(thrust::norm(flat_constr_data[i % (num / batch_size)]) != 0)
            flat_src_data[i] = flat_constr_data[i % (num / batch_size)];
    }
}

template<typename T>
__global__ void ker_MergeAddData(thrust::complex<T>* flat_src, thrust::complex<T>* flat_dst, T alpha, T beta, uint64_t num)
{
    for(uint64_t i = cg::this_grid().thread_rank(); i < num; i+= cg::this_grid().size())
    {
        flat_dst[i] = flat_src[i] * alpha + flat_dst[i] * beta;
    }
}

template<typename T>
__global__ void ker_Normalization(thrust::complex<T>* flat_src, T norm, uint64_t num)
{
    for(uint64_t i = cg::this_grid().thread_rank(); i < num; i+= cg::this_grid().size())
    {
        flat_src[i] /= norm;
    }
}


}
}