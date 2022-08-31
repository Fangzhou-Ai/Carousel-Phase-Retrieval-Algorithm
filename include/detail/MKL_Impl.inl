#pragma once
#include <random>

namespace CPRA
{

template<typename T>
bool MklImpl<T>::Initialize(T* flat_data_ptr, size_t num)
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
    for(size_t i = 0; i < num; i++)
    {
        flat_data_ptr[i] = distribution(generator);
    }
#ifdef HAS_OMP
}
#endif
    return true;
}

template<typename T>
bool MklImpl<T>::MergeAddData(std::complex<T>* flat_src, std::complex<T>* flat_dst, T alpha, T beta, size_t num)
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
bool MklImpl<T>::Normalization(std::complex<T>* flat_src, T norm, size_t num)
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
bool MklImpl<T>::SpaceConstraint(std::complex<T>* flat_src_data, T* flat_constr_data, size_t num, size_t batch_size)
{
    if(num == 0 || batch_size == 0) return false;
#ifdef HAS_OMP
#pragma omp parallel for
#endif
    for(size_t i = 0; i < num; i++)
    {
        flat_src_data[i].imag(0);
        if(flat_constr_data[i % (num / batch_size)] == 0)
            flat_src_data[i].real(0);
    }
    return true;
}

template<typename T>
bool MklImpl<T>::RealDataConstraint(std::complex<T>* flat_src_data, T* flat_constr_data, size_t num, size_t batch_size)
{
    if(num == 0 || batch_size == 0) return false;
#ifdef HAS_OMP
#pragma omp parallel for
#endif
    for(size_t i = 0; i < num; i++)
    {
        if(flat_constr_data[i % (num / batch_size)] == 0)
        {
            // unconstraint
            // we can rarely have 0 intensity in real case
            // therefore we set 0 intensity pixel as unconstraint
            continue;
        }
        if(std::norm(flat_src_data[i]) == 0)
            flat_src_data[i].real(flat_constr_data[i % (num / batch_size)]);
        else
            flat_src_data[i] = std::polar(flat_constr_data[i % (num / batch_size)], std::arg(flat_src_data[i]));
    }
    return true;
}

template<typename T>
bool MklImpl<T>::ComplexDataConstraint(std::complex<T>* flat_src_data, std::complex<T>* flat_constr_data, size_t num, size_t batch_size)
{

    if(num == 0 || batch_size == 0) return false;
#ifdef HAS_OMP
#pragma omp parallel for
#endif
    for(size_t i = 0; i < num; i++)
    {
        if(std::norm(flat_constr_data[i % (num / batch_size)]) == 0)
        {
            // unconstraint
            // we can rarely have 0 intensity in real case
            // therefore we set 0 intensity pixel as unconstraint
            continue;
        }
        else
            flat_src_data[i] = flat_constr_data[i % (num / batch_size)];
    }
    return true;
}

template<typename T>
bool MklImpl<T>::Forward2D(std::complex<T>* flat_input)
{ 
    status = DftiComputeForward(Dfti2DHandle_, flat_input);
    return true;
}

template<typename T>
bool MklImpl<T>::Backward2D(std::complex<T>* flat_input)
{
    status = DftiComputeBackward(Dfti2DHandle_, flat_input);
    return true;
}

template<typename T>
bool MklImpl<T>::Forward3D(std::complex<T>* flat_input)
{
    status = DftiComputeForward(Dfti3DHandle_, flat_input);
    return true;
}

template<typename T>
bool MklImpl<T>::Backward3D(std::complex<T>* flat_input)
{
    status = DftiComputeBackward(Dfti3DHandle_, flat_input);
    return true;
}


} // namespace CPRA