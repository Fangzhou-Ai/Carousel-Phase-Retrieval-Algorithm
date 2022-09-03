#pragma once
#include <memory>
#include <filesystem>
#include <cstddef>
#include <string>
#include "Impl_Factory.hpp"
#include "Allocator.hpp"

namespace CPRA{

template<typename T, IMPL_TYPE type>
class Cpra
{
    public:
        Cpra() = default;
        Cpra(size_t m, size_t n, size_t l, size_t batch_size) : type_(type)
        {
           impl_ = NewCpraImpl<T>(type_, m, n, l, batch_size);
           alloc_ = NewAllocator(type_);
        }

        ~Cpra(){}
        // Data I/O  
        bool ReadMatrixFromFile(std::string FileName, T* flat_data_ptr, size_t m, size_t n, size_t l);
        bool WriteMatrixToFile(std::string FileName, T* flat_data_ptr, size_t m, size_t n, size_t l);

        // Reconstruction
        bool PreReconstruct(size_t iter_num, size_t batch_size = 1){};
        bool Reconstruct(size_t epi, size_t iter_per_epi, size_t batch_size = 1){};

        // mem alloc
        void* allocate(size_t alloc_bytes, int alignment = 64)
        {
            return alloc_->allocate(alloc_bytes, alignment);
        }

        void deallocate(void* dealloc_ptr)
        {
            alloc_->deallocate(dealloc_ptr);
        }
        
        std::unique_ptr<CpraImpl<T>> impl_;
    private:  
        std::unique_ptr<AllocatorInterface> alloc_;
        const IMPL_TYPE type_;
};



} // namespace CPRA

#include "./detail/CPRA.inl"