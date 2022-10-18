#pragma once
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cuda.h>
#include <complex>
#include <thrust/swap.h>

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


template <typename T>
__global__ void ker_FFTShift2D(thrust::complex<T>* flat_input, size_t M, size_t N, size_t batch_size)
{
    // Assume that M and N are even
    for(uint64_t i = cg::this_grid().thread_rank(); i < (uint64_t)M * N * batch_size / 2; i+= cg::this_grid().size())
    {
        uint64_t bidx = i / ((uint64_t)M * N / 2);
        uint64_t idx = i - bidx * M * N / 2;
        size_t col = idx % N; // [0, N - 1]
        size_t row = idx / N; // [0, M / 2 - 1]
        uint64_t offset = bidx * M * N;
        size_t crow = row + M / 2;
        size_t ccol;
        // which squard
        if(col < N / 2)
        {
            // first quad to third quad
            ccol = col + N / 2;  
        }
        else
        {
            // second quad to forth quard
            ccol = col - N / 2;
        }
        thrust::swap(flat_input[crow * N + ccol + offset], flat_input[row * N + col + offset]);
    }
}

template <typename T>
__global__ void ker_FFTShift3D(thrust::complex<T>* flat_input, size_t M, size_t N, size_t L, size_t batch_size)
{
    // Assume that M, N, L are even
    for(uint64_t i = cg::this_grid().thread_rank(); i < (uint64_t)M * N * L * batch_size / 2; i+= cg::this_grid().size())
    {
        uint64_t bidx = i / ((uint64_t)M * N * L / 2);
        uint64_t idx = i - bidx * M * N * L / 2;
        size_t  dep = idx / (M * N);
        idx = idx - dep * M * N;
        size_t col = idx % N;
        size_t row = idx / N;
        auto offset  = bidx * M * N * L;
        size_t cdep = dep + L / 2;
        size_t crow;
        size_t ccol;

        if(row < M / 2)
        {
            crow = row + M / 2;
        }
        else
        {
            crow = row - M / 2;
        }

        if(col < N / 2)
        {
            ccol = col + N / 2;
        }
        else
        {
            ccol = col - N / 2;
        }
        thrust::swap(flat_input[cdep * M * N + crow * N + ccol + offset], flat_input[dep * M * N + row * N + col + offset]);
    }
}


}
}