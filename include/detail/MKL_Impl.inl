#pragma once
#include <random>

namespace CPRA
{

template<typename T>
bool MklImpl<T>::Initialize(T* flat_data_ptr, size_t m, size_t n, size_t l)
{
    // Check params
    if(m < 1) m = 1;
    if(n < 1) n = 1;
    if(l < 1) l = 1;

    std::mt19937 generator;
    std::uniform_real_distribution<T> distribution(0, 1);
    for(size_t i = 0; i < m * n * l; i++)
    {
        flat_data_ptr[i] = distribution(generator);
    }
    return true;
}

}