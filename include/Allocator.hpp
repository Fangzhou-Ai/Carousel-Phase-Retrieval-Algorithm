#include "CPRA_Impl.hpp"

namespace CPRA{

class AllocatorInterface
{
    public:
        AllocatorInterface(){}
        virtual ~AllocatorInterface(){}

        virtual void* allocate(size_t alloc_bytes, int alignment = 64) const = 0;
        virtual void deallocate(void* dealloc_ptr) const = 0;
};


#if HAS_CUDA

class CudaAllocator final : public AllocatorInterface
{
    public:
        CudaAllocator(){}
        ~CudaAllocator(){}
        void* allocate(size_t alloc_bytes, int alignment = 64) const override
        {
            void* ptr;
            cudaMalloc((void**) & ptr, alloc_bytes);
            return ptr;
        }

        void deallocate(void* dealloc_ptr) const override
        {
            cudaFree(dealloc_ptr);
        }
};

class CudaManagedAllocator final : public AllocatorInterface
{
    public:
        CudaManagedAllocator(){}
        ~CudaManagedAllocator(){}
        void* allocate(size_t alloc_bytes, int alignment = 64) const override
        {
            void* ptr;
            cudaMallocManaged((void**) & ptr, alloc_bytes);
            return ptr;
        }

        void deallocate(void* dealloc_ptr) const override
        {
            cudaFree(dealloc_ptr);
        }
};
#endif

#ifdef HAS_MKL
class MklAllocator final : public AllocatorInterface
{
    public:
        MklAllocator(){}
        ~MklAllocator(){}
        void* allocate(size_t alloc_bytes, int alignment = 64) const override
        {
            return mkl_malloc(alloc_bytes, alignment);
        }

        void deallocate(void* dealloc_ptr) const override
        {
            mkl_free(dealloc_ptr);
        }
};
#endif

std::unique_ptr<AllocatorInterface> NewAllocator(IMPL_TYPE type, bool gpu_managed = true)
{
    switch(type)
    {
#ifdef HAS_CUDA
        case IMPL_TYPE::CUDA:
            if(gpu_managed)
                return std::make_unique<CudaManagedAllocator>();
            else
                return std::make_unique<CudaAllocator>();
            break;
#endif
#ifdef HAS_MKL
        case IMPL_TYPE::MKL:
            return std::make_unique<MklAllocator>();
            break;
#endif
        default:
            throw std::invalid_argument("Wrong allocator type! Only support CUDA and MKL for now.");
    }
}



}