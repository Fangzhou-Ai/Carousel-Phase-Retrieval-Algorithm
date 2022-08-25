#pragma once
#ifdef HAS_CUDA
#include "CPRA_Interface.hpp"
#include "CUDA_Error_Check.hpp"

namespace CPRA{
template <typename T>
class CudaCpra final : public CpraInterface<T>
{
    public:
        CudaCpra(){}

        // Initialize
        virtual bool Initialize(T* flat_data_ptr, size_t m, size_t n, size_t l) override;
        // Reconstruction
        virtual bool PreReconstruct(size_t iter_num, size_t batch_size = 1) override;
        virtual bool Reconstruct(size_t epi, size_t iter_per_epi, size_t batch_size = 1) override;

        ~CudaCpra(){}
}; // CudaCpra

} // CPRA

#include "./detail/CUDA_CPRA.inl"

#endif // HAS_CUDA