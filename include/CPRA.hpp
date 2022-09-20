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
        Cpra(uint64_t m, uint64_t n, uint64_t l, uint64_t batch_size) : type_(type)
        {
           impl_ = NewCpraImpl<T>(type_, m, n, l, batch_size);
           alloc_ = NewAllocator(type_);
        }

        ~Cpra(){}
        // Data I/O  
        bool ReadMatrixFromFile(std::string FileName, T* flat_data_ptr, uint64_t m, uint64_t n, uint64_t l);
        bool WriteMatrixToFile(std::string FileName, T* flat_data_ptr, uint64_t m, uint64_t n, uint64_t l);

        // Reconstruction
        bool PreReconstruct(uint64_t iter_num, uint64_t batch_size = 1){};
        bool Reconstruct(uint64_t epi, uint64_t iter_per_epi, uint64_t batch_size = 1){};
        // convert data type
        bool RealToComplex(T* flat_src, std::complex<T>* flat_dst, uint64_t num)
        {
            for(auto i = 0; i < num; i++)
            {
                flat_dst[i].real(flat_src[i]);
                flat_dst[i].imag(0);
            }
            return true;
        }
        bool ComplexToReal(std::complex<T>* flat_src, T* flat_dst, uint64_t num)
        {
            for(auto i = 0; i < num; i++)
            {
                flat_dst[i] = flat_src[i].real();
            }
            return true;
        }
        // mem alloc
        void* allocate(uint64_t alloc_bytes, int alignment = 64)
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