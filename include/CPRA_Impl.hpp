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
#ifdef HAS_MKL
        // MKL version, without cuda stream here
        virtual bool Forward2D(T* flat_input, T* flat_output, size_t m, size_t n, size_t batch_size, bool inplace = false) = 0;

        virtual bool Backward2D(T* flat_input, T* flat_output, size_t m, size_t n, size_t batch_size, bool inplace = false) = 0;

        virtual bool Forward3D(T* flat_input, T* flat_output, size_t m, size_t n, size_t l, size_t batch_size, bool inplace = false) = 0;

        virtual bool Backward3D(T* flat_input, T* flat_output, size_t m, size_t n, size_t l, size_t batch_size, bool inplace = false) = 0;

        virtual bool SpaceConstraint(T* flat_src_data, bool* flat_constr_data, size_t num, size_t batch_size) = 0;

        virtual bool RealDataConstraint(std::complex<T>* flat_src_data, T* flat_constr_data, size_t num, size_t batch_size) = 0;

        virtual bool ComplexDataConstraint(std::complex<T>* flat_src_data, std::complex<T>* flat_constr_data, size_t num, size_t batch_size) = 0;

        // Add src to dst
        // flat_dst = alpha * flat_src + flat_dst 
        virtual bool MergeAddData(T* flat_src, T* flat_dst, T alpha, size_t num) = 0;

        // flat_src = flat_src ./ norm
        virtual bool Normalization(T* flat_src, T norm, size_t num) = 0;
#endif
#ifdef HAS_CUDA
        // CUDA version with cuda stream here
        virtual bool Forward2D(T* flat_input, T* flat_output, size_t m, size_t n, size_t batch_size, cudaStream_t stream = 0,  bool inplace = false) = 0;

        virtual bool Backward2D(T* flat_input, T* flat_output, size_t m, size_t n, size_t batch_size, cudaStream_t stream = 0,  bool inplace = false) = 0;

        virtual bool Forward3D(T* flat_input, T* flat_output, size_t m, size_t n, size_t l, size_t batch_size, cudaStream_t stream = 0,  bool inplace = false) = 0;

        virtual bool Backward3D(T* flat_input, T* flat_output, size_t m, size_t n, size_t l, size_t batch_size, cudaStream_t stream = 0,  bool inplace = false) = 0;

        virtual bool SpaceConstraint(T* flat_src_data, bool* flat_constr_data, size_t num, size_t batch_size, cudaStream_t stream = 0) = 0;

        virtual bool RealDataConstraint(std::complex<T>* flat_src_data, T* flat_constr_data, size_t num, size_t batch_size, cudaStream_t stream = 0) = 0;

        virtual bool ComplexDataConstraint(std::complex<T>* flat_src_data, std::complex<T>* flat_constr_data, size_t num, size_t batch_size, cudaStream_t stream = 0) = 0;

        // Add src to dst
        // flat_dst = alpha * flat_src + flat_dst 
        virtual bool MergeAddData(T* flat_src, T* flat_dst, T alpha, size_t num, cudaStream_t stream = 0) = 0;

        // flat_src = flat_src ./ norm
        virtual bool Normalization(T* flat_src, T norm, size_t num, cudaStream_t stream = 0) = 0;

#endif
        // Only support one rotating angle for now
        // param:
        // p : number of 2D sources
        // m, n, l: 3 dimensions
        virtual bool Real2DTo3DInterpolation(T* flat_2d_src, T* flat_3D_dst, T* angles, size_t m, size_t n, size_t p, size_t l) = 0;

        virtual bool Complex2DTo3DInterpolation(std::complex<T>* flat_2d_src, std::complex<T>* flat_3D_dst, T* angles, size_t m, size_t n, size_t p, size_t l) = 0;

        virtual ~CpraImpl(){}
};


} // name space CPRA