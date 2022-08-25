#pragma once
#ifdef HAS_CUDA
#include "CPRA_Impl.hpp"
#include "CUDA_Error_Check.hpp"

namespace CPRA{
template <typename T>
class CudaImpl final : public CpraImpl<T>
{
    public:
        CudaImpl(){}

        // Initialize
        bool Initialize(T* flat_data_ptr, size_t m, size_t n, size_t l) override;

        ~CudaImpl(){}
}; // CudaCpra

template<typename T>
std::unique_ptr<CpraImpl<T>> NewCUDAImpl()
{
    return std::make_unique<CudaImpl<T>>();
}

} // namespace CPRA

#include "./detail/CUDA_Impl.inl"

#endif // HAS_CUDA