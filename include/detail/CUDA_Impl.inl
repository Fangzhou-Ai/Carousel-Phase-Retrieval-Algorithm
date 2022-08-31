#pragma once
#include <curand.h>
#include <time.h>
#include <typeinfo>

namespace CPRA{

template<typename T>
bool CudaImpl<T>::Initialize(T* flat_data_ptr, size_t num)
{
    // Check params
    if(num < 1) return false;

    // Random Uniform
    curandGenerator_t handle;
    CPRA_CURAND_CALL(curandCreateGenerator(&handle, CURAND_RNG_PSEUDO_MT19937));
    CPRA_CURAND_CALL(curandSetPseudoRandomGeneratorSeed(handle, time(NULL)));
    if constexpr(std::is_same_v<T, float>)
        CPRA_CURAND_CALL(curandGenerateUniform(handle, flat_data_ptr, num));
    else
        CPRA_CURAND_CALL(curandGenerateUniformDouble (handle, flat_data_ptr, num));
    CPRA_CURAND_CALL(curandDestroyGenerator(handle));
    return true;
}

template<typename T>
bool CudaImpl<T>::Forward2D(thrust::complex<T>* flat_input, cudaStream_t stream)
{
    return true;
}

}