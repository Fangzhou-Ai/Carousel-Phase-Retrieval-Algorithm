#pragma once

#ifdef HAS_CUDA
#include "CUDA_Impl.cuh"
#endif

#ifdef HAS_MKL
#include "MKL_Impl.hpp"
#endif

namespace CPRA{

template <typename T>
std::unique_ptr<CpraImpl<T>> NewCpraImpl(IMPL_TYPE type, uint64_t m, uint64_t n, uint64_t l, uint64_t batch_size)
{
    switch(type)
    {
#ifdef HAS_CUDA
        case IMPL_TYPE::CUDA:
            return NewCUDAImpl<T>(m, n, l, batch_size);
            break;
#endif
#ifdef HAS_MKL
        case IMPL_TYPE::MKL:
            return NewMKLImpl<T>(m, n, l, batch_size);
            break;
#endif
        default:
            throw std::invalid_argument("Wrong impl type! Only support CUDA and MKL for now.");
    }
}

} // namespace CPRA