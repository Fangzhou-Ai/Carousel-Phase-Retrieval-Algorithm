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
bool MklImpl<T>::MergeAddData(T* flat_src, T* flat_dst, T alpha, size_t num)
{
    if(alpha == 0) return true;
#ifdef HAS_OMP
#pragma omp parallel for
#endif
    for(size_t i = 0; i < num; i++)
    {
        flat_dst[i] += alpha * flat_src[i];
    }
}

template<typename T>
bool MklImpl<T>::Normalization(T* flat_src, T norm, size_t num)
{
    if(norm == 0) return false;
#ifdef HAS_OMP
#pragma omp parallel for
#endif
    for(size_t i = 0; i < num; i++)
    {
        flat_src[i] /= norm;
    }
}

template<typename T>
bool MklImpl<T>::SpaceConstraint(T* flat_src_data, bool* flat_constr_data, size_t num, size_t batch_size)
{
    if(num == 0 || batch_size == 0) return false;
#ifdef HAS_OMP
#pragma omp parallel for
#endif
    for(size_t i = 0; i < num; i++)
    {
        if(!flat_constr_data[i % (num / batch_size)])
            flat_src_data[i] = 0; 
    }
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
}



} // namespace CPRA