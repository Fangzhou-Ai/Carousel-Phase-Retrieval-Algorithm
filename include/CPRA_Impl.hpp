#pragma once
#ifdef HAS_CUDA
#include <cuda.h>
#include <cufft.h>
#endif

#ifdef HAS_MKL
#include <mkl.h>
#endif

#include <string>
#include <complex>

namespace CPRA{
enum IMPL_TYPE {CUDA, MKL};


template<typename T>
class CpraImpl
{
    public:
        CpraImpl(){}
        virtual bool Initialize(T* flat_data_ptr, size_t num) = 0;
        // Fixed FFT size for better performance
        virtual bool Forward2D(std::complex<T>* flat_input) = 0;

        virtual bool Backward2D(std::complex<T>* flat_input) = 0;

        virtual bool Forward3D(std::complex<T>* flat_input) = 0;

        virtual bool Backward3D(std::complex<T>* flat_input) = 0;

        virtual bool SpaceConstraint(std::complex<T>* flat_src_data, T* flat_constr_data, size_t num, size_t batch_size) = 0;

        virtual bool RealDataConstraint(std::complex<T>* flat_src_data, T* flat_constr_data, size_t num, size_t batch_size) = 0;

        virtual bool ComplexDataConstraint(std::complex<T>* flat_src_data, std::complex<T>* flat_constr_data, size_t num, size_t batch_size) = 0;

        // Add src to dst
        // flat_dst = alpha * flat_src + flat_dst 
        virtual bool MergeAddData(std::complex<T>* flat_src, std::complex<T>* flat_dst, T alpha, T beta, size_t num) = 0;

        // flat_src = flat_src ./ norm
        virtual bool Normalization(std::complex<T>* flat_src, T norm, size_t num) = 0;
        // Only support one rotating angle for now
        // param:
        // p : number of 2D sources
        // m, n, l: 3 dimensions
        // To interpolate real value, cast it to complex first
        virtual bool Complex2DTo3DInterpolation(std::complex<T>* flat_2d_src, std::complex<T>* flat_3D_dst, T* angles, size_t m, size_t n, size_t p, size_t l) = 0;

        virtual bool Memcpy(void* dst, void* src, size_t bytes) = 0;
        
        virtual bool Sync() = 0;
        
        virtual ~CpraImpl(){}
};


} // name space CPRA