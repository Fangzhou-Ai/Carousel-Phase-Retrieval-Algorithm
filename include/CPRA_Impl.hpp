#pragma once
#ifdef HAS_CUDA
#include <cuda.h>
#include <cufft.h>
#endif

#ifdef HAS_MKL
#include <mkl.h>
#endif

#include <string>


namespace CPRA{
enum IMPL_TYPE {CUDA, MKL};


template<typename T>
class CpraImpl
{
    public:
        CpraImpl(){}
        virtual bool Initialize(T* flat_data_ptr, size_t m, size_t n, size_t l) = 0;
        virtual ~CpraImpl(){}
};


} // name space CPRA