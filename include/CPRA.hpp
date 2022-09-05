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
        Cpra(unsigned long long m, unsigned long long n, unsigned long long l, unsigned long long batch_size) : type_(type)
        {
           impl_ = NewCpraImpl<T>(type_, m, n, l, batch_size);
           alloc_ = NewAllocator(type_);
        }

        ~Cpra(){}
        // Data I/O  
        bool ReadMatrixFromFile(std::string FileName, T* flat_data_ptr, unsigned long long m, unsigned long long n, unsigned long long l);
        bool WriteMatrixToFile(std::string FileName, T* flat_data_ptr, unsigned long long m, unsigned long long n, unsigned long long l);

        // Reconstruction
        bool PreReconstruct(unsigned long long iter_num, unsigned long long batch_size = 1){};
        bool Reconstruct(unsigned long long epi, unsigned long long iter_per_epi, unsigned long long batch_size = 1){};

        // mem alloc
        void* allocate(unsigned long long alloc_bytes, int alignment = 64)
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