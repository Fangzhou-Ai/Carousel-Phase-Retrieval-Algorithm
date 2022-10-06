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
        if(flat_src_data[i].real() < 0)
            flat_src_data[i].real(0);
        else
            flat_src_data[i].real(flat_src_data[i].real() * flat_constr_data[i % (num / batch_size)]);
    }
}

template<typename T>
__global__ void ker_DataConstraint(thrust::complex<T>* flat_src_data, T* flat_constr_data, uint64_t num, uint64_t batch_size)
{
    for(uint64_t i = cg::this_grid().thread_rank(); i < num; i+= cg::this_grid().size())
    {
        if(thrust::norm(flat_src_data[i]) > 0)
            flat_src_data[i] = thrust::polar<T, T>(flat_constr_data[i % (num / batch_size)], thrust::arg(flat_src_data[i]));
    }
}

template<typename T>
__global__ void ker_DataConstraint(thrust::complex<T>* flat_src_data, thrust::complex<T>* flat_constr_data, uint64_t num, uint64_t batch_size)
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

template <typename T>
__global__ void ker_Real2DTo3DInterpolation(T* flat_2d_src, T* flat_3d_dst, T* angles, T* flat_weight, uint64_t M, uint64_t N, uint64_t P, uint64_t L)
{
    for(uint64_t i = cg::this_grid().thread_rank(); i < M * N * P; i+= cg::this_grid().size())
    {
        //in 2D slices coordinate, follow Matlab convention
        int zIn = i / (M * N);
        int xIn = (i - M * N * zIn) / M;
        int yIn = (i - M * N * zIn) - M * xIn;
        //convert to 3D Euclidean space coordinate
        T xReal = (T)xIn - (T)N / 2.0;
        T yReal = (T)-yIn + (T)M / 2.0;
        T ang = angles[zIn];
        T x3D = yReal * sin(ang);
        T y3D = xReal;
        T z3D = yReal * cos(ang);
        T distance;
        int j = round(x3D);
        int k = round(y3D);
        int l = round(z3D);
        //convert to 3D matrix coordinate
        int z = L / 2 - j;
        int x = N / 2 + k;
        int y = M / 2 - l;
        int index = z * M * N + x * M + y;
        distance = sqrt(pow(x3D - j, 2) + pow(y3D - k, 2) + pow(z3D - l, 2));
        if (index >= M * N * L || index < 0 || distance >= 0.5)
            return;

        if (abs(distance) <= 0.02)
        {
            flat_weight[index] += 50;
            flat_3d_dst[index] = flat_3d_dst[index] + 50 * flat_2d_src[i];
        }
        else
        {
            flat_weight[index] += 1.0 / distance;
            flat_3d_dst[index] = flat_3d_dst[index] + flat_2d_src[i] * (1.0 / distance);
        }
    }
}


template <typename T>
__global__ void ker_NormRealInterpolation(T* flat_3d_dst, T* flat_weight, uint64_t num)
{
    for(uint64_t i = cg::this_grid().thread_rank(); i < num; i+= cg::this_grid().size())
    {
        if(flat_weight[i] > 0)
            flat_3d_dst[i] /= flat_weight[i];
    }
}



}
}