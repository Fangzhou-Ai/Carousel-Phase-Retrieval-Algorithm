#pragma once
#include <time.h>
#include <typeinfo>
#include "./detail/CUDA_Impl_Kernel.cuh"

namespace CPRA{

template<typename T>
bool CudaImpl<T>::Initialize(T* flat_data_ptr, uint64_t num)
{
    // Check params
    if(num < 1) return false;
    if constexpr(std::is_same_v<T, float>)
        CPRA_CURAND_CALL(curandGenerateUniform(cuRandHandle, flat_data_ptr, num));
    else
        CPRA_CURAND_CALL(curandGenerateUniformDouble (cuRandHandle, flat_data_ptr, num));
    return true;
}

template<typename T>
bool CudaImpl<T>::Forward2D(std::complex<T>* flat_input)
{
    if constexpr (std::is_same_v<T, float>)
    {
        cufftExecC2C(Dfti2DHandle_, (cuComplex*)flat_input, (cuComplex*)flat_input, CUFFT_FORWARD);
    }
    else
    {
        cufftExecZ2Z(Dfti2DHandle_, (cuDoubleComplex*)flat_input, (cuDoubleComplex*)flat_input, CUFFT_FORWARD);
    }
    return true;
}

template<typename T>
bool CudaImpl<T>::Backward2D(std::complex<T>* flat_input)
{
    if constexpr (std::is_same_v<T, float>)
    {
        cufftExecC2C(Dfti2DHandle_, (cuComplex*)flat_input, (cuComplex*)flat_input, CUFFT_INVERSE);
    }
    else
    {
        cufftExecZ2Z(Dfti2DHandle_, (cuDoubleComplex*)flat_input, (cuDoubleComplex*)flat_input, CUFFT_INVERSE);
    }
    Normalization(flat_input, m_ * n_, m_ * n_ * batch_);
    return true;
}

template<typename T>
bool CudaImpl<T>::Forward3D(std::complex<T>* flat_input)
{
    if constexpr (std::is_same_v<T, float>)
    {
        cufftExecC2C(Dfti3DHandle_, (cuComplex*)flat_input, (cuComplex*)flat_input, CUFFT_FORWARD);
    }
    else
    {
        cufftExecZ2Z(Dfti3DHandle_, (cuDoubleComplex*)flat_input, (cuDoubleComplex*)flat_input, CUFFT_FORWARD);
    }
    return true;
}

template<typename T>
bool CudaImpl<T>::Backward3D(std::complex<T>* flat_input)
{
    if constexpr (std::is_same_v<T, float>)
    {
        cufftExecC2C(Dfti3DHandle_, (cuComplex*)flat_input, (cuComplex*)flat_input, CUFFT_INVERSE);
    }
    else
    {
        cufftExecZ2Z(Dfti3DHandle_, (cuDoubleComplex*)flat_input, (cuDoubleComplex*)flat_input, CUFFT_INVERSE);
    }
    Normalization(flat_input, m_ * n_ * l_, m_ * n_* l_ * batch_);
    return true;
}

template<typename T>
bool CudaImpl<T>::SpaceConstraint(std::complex<T>* flat_src_data, T* flat_constr_data, uint64_t num, uint64_t batch_size)
{
    uint64_t block_size = 256;
    uint64_t per_thread_data = 8;
    uint64_t per_block_data = block_size * per_thread_data;
    uint64_t grid_size = (num + per_block_data - 1) / per_block_data;
    Kernel::ker_SpaceConstraint<T><<<grid_size, block_size, 0, stream_>>>
    ((thrust::complex<T>*)flat_src_data, flat_constr_data, num, batch_size);
    return true;
}

template<typename T>
bool CudaImpl<T>::DataConstraint(std::complex<T>* flat_src_data, T* flat_constr_data, uint64_t num, uint64_t batch_size)
{
    uint64_t block_size = 256;
    uint64_t per_thread_data = 8;
    uint64_t per_block_data = block_size * per_thread_data;
    uint64_t grid_size = (num + per_block_data - 1) / per_block_data;
    Kernel::ker_DataConstraint<T><<<grid_size, block_size, 0, stream_>>>
    ((thrust::complex<T>*)flat_src_data, flat_constr_data, num, batch_size);
    return true;
}

template<typename T>
bool CudaImpl<T>::DataConstraint(std::complex<T>* flat_src_data, std::complex<T>* flat_constr_data, uint64_t num, uint64_t batch_size)
{
    uint64_t block_size = 256;
    uint64_t per_thread_data = 8;
    uint64_t per_block_data = block_size * per_thread_data;
    uint64_t grid_size = (num + per_block_data - 1) / per_block_data;
    Kernel::ker_DataConstraint<T><<<grid_size, block_size, 0, stream_>>>
    ((thrust::complex<T>*)flat_src_data, (thrust::complex<T>*)flat_constr_data, num, batch_size);
    return true;
}

template<typename T>
bool CudaImpl<T>::MergeAddData(std::complex<T>* flat_src, std::complex<T>* flat_dst, T alpha, T beta, uint64_t num)
{
    uint64_t block_size = 256;
    uint64_t per_thread_data = 8;
    uint64_t per_block_data = block_size * per_thread_data;
    uint64_t grid_size = (num + per_block_data - 1) / per_block_data;
    Kernel::ker_MergeAddData<T><<<grid_size, block_size, 0, stream_>>>
    ((thrust::complex<T>*)flat_src, (thrust::complex<T>*)flat_dst, alpha, beta, num);
    return true;
}

template<typename T>
bool CudaImpl<T>::Normalization(std::complex<T>* flat_src, T norm, uint64_t num)
{
    uint64_t block_size = 256;
    uint64_t per_thread_data = 8;
    uint64_t per_block_data = block_size * per_thread_data;
    uint64_t grid_size = (num + per_block_data - 1) / per_block_data;
    Kernel::ker_Normalization<T><<<grid_size, block_size, 0, stream_>>>
    ((thrust::complex<T>*)flat_src, norm, num);
    return true;
}
template<typename T>
bool CudaImpl<T>:: ConvergeError(std::complex<T>* flat_old, 
                                 std::complex<T>* flat_new,
                                 T* flat_error,
                                 uint64_t m, uint64_t n, uint64_t l, uint64_t batch_size)
{
    MergeAddData(flat_new, flat_old, 1, -1, m * n * l * batch_size);
    T TErr = 0;
    if constexpr(std::is_same_v<T, float>)
    {
        for(auto i = 0; i < batch_size; i++)
        {
            CPRA_CUBLAS_CALL(cublasScnrm2(cuBlasHandle, m * n, reinterpret_cast<cuComplex*>(flat_old) + i * m * n * l, 1, flat_error + i));
            CPRA_CUBLAS_CALL(cublasScnrm2(cuBlasHandle, m * n, reinterpret_cast<cuComplex*>(flat_new) + i * m * n * l, 1, &TErr));
            flat_error[i] /= TErr;
        }
    }
    else
    {
        for(auto i = 0; i < batch_size; i++)
        {
            CPRA_CUBLAS_CALL(cublasDznrm2(cuBlasHandle, m * n, reinterpret_cast<cuDoubleComplex*>(flat_old) + i * m * n * l, 1, flat_error + i));
            CPRA_CUBLAS_CALL(cublasDznrm2(cuBlasHandle, m * n, reinterpret_cast<cuDoubleComplex*>(flat_new) + i * m * n * l, 1, &TErr));
			flat_error[i] /= TErr;
        }
    }
    return true;
}

template<typename T>
bool CudaImpl<T>::Real2DTo3DInterpolation(T* flat_2d_src, T* flat_3d_dst, T* angles, uint64_t m, uint64_t n, uint64_t p, uint64_t l)
{
    // Rotate around a flix axis, euler angle is not supported now
	// in this case it's rotating around y axis, from +z to +x to -z
    T* flat_weight;
    CPRA_CUDA_TRY(cudaMallocManaged((void**) & flat_weight, sizeof(T) * m * n * l));
    CPRA_CUDA_TRY(cudaMemsetAsync(flat_3d_dst, 0, sizeof(T) * m * n * l, stream_));
    CPRA_CUDA_TRY(cudaMemsetAsync(flat_weight, 0, sizeof(T) * m * n * l, stream_));
    
    uint64_t block_size = 256;
    uint64_t per_thread_data = 8;
    uint64_t per_block_data = block_size * per_thread_data;
    uint64_t grid_size = (m * n * p + per_block_data - 1) / per_block_data;
    Kernel::ker_Real2DTo3DInterpolation<T><<<grid_size, block_size, 0, stream_>>>
    (flat_2d_src, flat_3d_dst, angles, flat_weight, m, n, p, l);
    
    grid_size = (m * n * l + per_block_data - 1) / per_block_data;
    Kernel::ker_NormRealInterpolation<T><<<grid_size, block_size, 0, stream_>>>
    (flat_3d_dst, flat_weight, m * n * l);

    CPRA_CUDA_TRY(cudaFree(flat_weight));
    return true;
}


}