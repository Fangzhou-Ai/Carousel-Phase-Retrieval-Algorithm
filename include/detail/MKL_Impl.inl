#pragma once
#include <random>
#include "mkl_dfti.h"
namespace CPRA
{

template<typename T>
bool MklImpl<T>::Initialize(T* flat_data_ptr, uint64_t num)
{
    // Check params
    if(num < 1) return false;
#ifdef HAS_OMP
#pragma omp parallel
{
#endif
    thread_local std::mt19937 generator;
    std::uniform_real_distribution<T> distribution(0, 1);
#ifdef HAS_OMP
#pragma omp for
#endif
    for(uint64_t i = 0; i < num; i++)
    {
        flat_data_ptr[i] = distribution(generator);
    }
#ifdef HAS_OMP
}
#endif
    return true;
}

template<typename T>
bool MklImpl<T>::MergeAddData(std::complex<T>* flat_src, std::complex<T>* flat_dst, T alpha, T beta, uint64_t num)
{
    if(alpha == 0 || num == 0) return false;
    std::complex<T> _a;
    std::complex<T> _b;
    _a.real(alpha);
    _a.imag(0);
    _b.real(beta);
    _b.imag(0);
    if constexpr(std::is_same_v<T, float>)
    {
        cblas_caxpby(num, &_a, flat_src, 1, &_b, flat_dst, 1);
    }
    else
    {
        cblas_zaxpby(num, &_a, flat_src, 1, &_b, flat_dst, 1);
    }
    return true;
}

template<typename T>
bool MklImpl<T>::Normalization(std::complex<T>* flat_src, T norm, uint64_t num)
{
    if(norm == 0 || num == 0) return false;
    std::complex<T> a;
    a.real(1.0 / norm);
    a.imag(0);
    if constexpr(std::is_same_v<T, float>)
    {
        cblas_cscal(num, &a, flat_src, 1);
    }
    else
    {
        cblas_zscal(num, &a, flat_src, 1);
    }
    return true;
}

template<typename T>
bool MklImpl<T>::SpaceConstraint(std::complex<T>* flat_src_data, T* flat_constr_data, uint64_t num, uint64_t batch_size)
{
    if(num == 0 || batch_size == 0) return false;
#ifdef HAS_OMP
#pragma omp parallel for
#endif
    for(uint64_t i = 0; i < num; i++)
    {
        flat_src_data[i].imag(0);
        if(flat_constr_data[i % (num / batch_size)] < 0.9)
            flat_src_data[i].real(0);
    }
    return true;
}

template<typename T>
bool MklImpl<T>::DataConstraint(std::complex<T>* flat_src_data, T* flat_constr_data, uint64_t num, uint64_t batch_size)
{
    if(num == 0 || batch_size == 0) return false;
#ifdef HAS_OMP
#pragma omp parallel for
#endif
    for(uint64_t i = 0; i < num; i++)
    {
        if(flat_constr_data[i % (num / batch_size)] == 0)
        {
            // unconstraint
            // we can rarely have 0 intensity in real case
            // therefore we set 0 intensity pixel as unconstraint
            continue;
        }
        if(std::norm(flat_src_data[i]) > 0)
            flat_src_data[i] = std::polar(flat_constr_data[i % (num / batch_size)], std::arg(flat_src_data[i]));
    }
    return true;
}

template<typename T>
bool MklImpl<T>::DataConstraint(std::complex<T>* flat_src_data, std::complex<T>* flat_constr_data, uint64_t num, uint64_t batch_size)
{

    if(num == 0 || batch_size == 0) return false;
#ifdef HAS_OMP
#pragma omp parallel for
#endif
    for(uint64_t i = 0; i < num; i++)
    {
        if(std::norm(flat_constr_data[i % (num / batch_size)]) != 0)
        {
            // we can rarely have 0 intensity in real case
            // therefore we set 0 intensity pixel as unconstraint
            flat_src_data[i] = flat_constr_data[i % (num / batch_size)];
        }
    }
    return true;
}

template<typename T>
bool MklImpl<T>::Forward2D(std::complex<T>* flat_input)
{ 
    FFTShift2D(flat_input, m_, n_, batch_size_);
    status = DftiComputeForward(Dfti2DHandle_, flat_input);
    if (status && !DftiErrorClass(status, DFTI_NO_ERROR))
    {
        printf("Error: %s\n", DftiErrorMessage(status));
    }
    FFTShift2D(flat_input, m_, n_, batch_size_);
    return true;
}

template<typename T>
bool MklImpl<T>::Backward2D(std::complex<T>* flat_input)
{
    FFTShift2D(flat_input, m_, n_, batch_size_);
    status = DftiComputeBackward(Dfti2DHandle_, flat_input);
    if (status && !DftiErrorClass(status, DFTI_NO_ERROR))
    {
        printf("Error: %s\n", DftiErrorMessage(status));
    }
    FFTShift2D(flat_input, m_, n_, batch_size_);
    return true;
}

template<typename T>
bool MklImpl<T>::Forward3D(std::complex<T>* flat_input)
{
    FFTShift3D(flat_input, m_, n_, l_, batch_size_);
    status = DftiComputeForward(Dfti3DHandle_, flat_input);
    if (status && !DftiErrorClass(status, DFTI_NO_ERROR))
    {
        printf("Error: %s\n", DftiErrorMessage(status));
    }
    FFTShift3D(flat_input, m_, n_, l_, batch_size_);
    return true;
}

template<typename T>
bool MklImpl<T>::Backward3D(std::complex<T>* flat_input)
{
    FFTShift3D(flat_input, m_, n_, l_, batch_size_);
    status = DftiComputeBackward(Dfti3DHandle_, flat_input);
    if (status && !DftiErrorClass(status, DFTI_NO_ERROR))
    {
        printf("Error: %s\n", DftiErrorMessage(status));
    }
    FFTShift3D(flat_input, m_, n_, l_, batch_size_);
    return true;
}

template<typename T>
bool MklImpl<T>::FFTShift2D(std::complex<T>* flat_input, size_t M, size_t N, size_t Batch)
{
#ifdef HAS_OMP
#pragma omp parallel for
#endif
    for(uint64_t i = 0; i < (uint64_t)M * N * Batch / 2; i++)
    {
        uint64_t bidx = i / ((uint64_t)M * N / 2);
        uint64_t idx = i - bidx * M * N / 2;
        size_t col = idx % N; // [0, N - 1]
        size_t row = idx / N; // [0, M / 2 - 1]
        uint64_t offset = bidx * M * N;
        size_t crow = row + M / 2;
        size_t ccol;
        // which squard
        if(col < N / 2)
        {
            // first quad to third quad
            ccol = col + N / 2;  
        }
        else
        {
            // second quad to forth quard
            ccol = col - N / 2;
        }
        std::swap(flat_input[crow * N + ccol + offset], flat_input[row * N + col + offset]);
    }
    return true;
}


template<typename T>
bool MklImpl<T>::FFTShift3D(std::complex<T>* flat_input, size_t M, size_t N, size_t L, size_t Batch)
{
#ifdef HAS_OMP
#pragma omp parallel for
#endif
    for(uint64_t i = 0; i < (uint64_t)M * N * L * Batch / 2; i++)
    {
        uint64_t bidx = i / ((uint64_t)M * N * L / 2);
        uint64_t idx = i - bidx * M * N * L / 2;
        size_t  dep = idx / (M * N);
        idx = idx - dep * M * N;
        size_t col = idx % N;
        size_t row = idx / N;
        auto offset  = bidx * M * N * L;
        size_t cdep = dep + L / 2;
        size_t crow;
        size_t ccol;

        if(row < M / 2)
        {
            crow = row + M / 2;
        }
        else
        {
            crow = row - M / 2;
        }

        if(col < N / 2)
        {
            ccol = col + N / 2;
        }
        else
        {
            ccol = col - N / 2;
        }
        std::swap(flat_input[cdep * M * N + crow * N + ccol + offset], flat_input[dep * M * N + row * N + col + offset]);
    }
    return true;
}




} // namespace CPRA