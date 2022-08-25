#include <curand.h>
#include <time.h>
#include <typeinfo>

namespace CPRA{

template<typename T>
bool CudaCpra<T>::Initialize(T* flat_data_ptr, size_t m, size_t n, size_t l)
{
    // Check params
    if(m < 1) m = 1;
    if(n < 1) n = 1;
    if(l < 1) l = 1;

    // Random Uniform
    curandGenerator_t handle;
    CPRA_CURAND_CALL(curandCreateGenerator(&handle, CURAND_RNG_PSEUDO_MT19937));
    CPRA_CURAND_CALL(curandSetPseudoRandomGeneratorSeed(handle, time(NULL)));
    if constexpr(std::is_same_v<T, float>)
        CPRA_CURAND_CALL(curandGenerateUniform(handle, flat_data_ptr, m * n * l));
    else
        CPRA_CURAND_CALL(curandGenerateUniformDouble (handle, flat_data_ptr, m * n * l));
    CPRA_CURAND_CALL(curandDestroyGenerator(handle));
    return true;
}

template <typename T>
bool CudaCpra<T>::PreReconstruct(size_t iter_num, size_t batch_size)
{
    return true;
}

template <typename T>
bool CudaCpra<T>::Reconstruct(size_t epi, size_t iter_per_epi, size_t batch_size)
{
    return true;
}
}