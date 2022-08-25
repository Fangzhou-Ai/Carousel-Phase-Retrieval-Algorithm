#pragma once
#ifdef HAS_MKL
#include "CPRA_Impl.hpp"
namespace CPRA
{
template <typename T>    
class MklImpl final : public CpraImpl<T>
{
    public:
        MklImpl(){}
        
        // Initialize
        bool Initialize(T* flat_data_ptr, size_t m, size_t n, size_t l) override;

        ~MklImpl(){}
};  // MklCpra

template<typename T>
std::unique_ptr<CpraImpl<T>> NewMKLImpl()
{
    return std::make_unique<MklImpl<T>>();
}

}// namespace CPRA

#include "./detail/MKL_Impl.inl"

#endif