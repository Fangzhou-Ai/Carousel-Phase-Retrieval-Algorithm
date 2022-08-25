#pragma once

#ifdef HAS_CUDA
#include "CUDA_Impl.cuh"
#endif

#ifdef HAS_MKL
#include "MKL_Impl.hpp"
#endif

namespace CPRA{

template <typename T>
std::unique_ptr<CpraImpl<T>> NewCpraImpl(IMPL_TYPE type)
{
    switch(type)
    {
#ifdef HAS_CUDA
        case IMPL_TYPE::CUDA:
            return NewCUDAImpl<T>();
            break;
#endif
#ifdef HAS_MKL
        case IMPL_TYPE::MKL:
            return NewMKLImpl<T>();
            break;
#endif
        default:
            throw std::invalid_argument("Wroong impl type! Only support CUDA and MKL for now.");
    }
}

} // namespace CPRA