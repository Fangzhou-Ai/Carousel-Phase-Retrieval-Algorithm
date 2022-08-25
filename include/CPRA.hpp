#pragma once
#include "Impl_Factory.hpp"

#include <filesystem>
#include <cstddef>
#include <string>

namespace CPRA{

template<typename T, IMPL_TYPE type = IMPL_TYPE::CUDA>
class Cpra
{
    public:
        Cpra() : type_(type)
        {
           impl_ = NewCpraImpl<T>(type);
        }

        ~Cpra(){}
        // Data I/O  
        bool ReadMatrixFromFile(std::string FileName, T* flat_data_ptr, size_t m, size_t n, size_t l);
        bool WriteMatrixToFile(std::string FileName, T* flat_data_ptr, size_t m, size_t n, size_t l);

        // Reconstruction
        bool PreReconstruct(size_t iter_num, size_t batch_size = 1){};
        bool Reconstruct(size_t epi, size_t iter_per_epi, size_t batch_size = 1){};
    private:
        std::unique_ptr<CpraImpl<T>> impl_;
        const IMPL_TYPE type_;
};



} // namespace CPRA

#include "./detail/CPRA.inl"