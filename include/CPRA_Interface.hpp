#ifdef HAS_CUDA
#include <cuda.h>
#include <cufft.h>
#endif

#ifdef HAS_MKL
#include <mkl.h>
#endif

#include <cstddef>
#include <string>
#include <filesystem>

namespace CPRA{

template<typename T>
class CpraInterface
{
    public:
        CpraInterface(){}

        virtual ~CpraInterface(){}
        // Data I/O  
        bool ReadMatrixFromFile(std::string FileName, T* flat_data_ptr, size_t m, size_t n, size_t l);
        bool WriteMatrixToFile(std::string FileName, T* flat_data_ptr, size_t m, size_t n, size_t l);
        // Initialize random unifor between 0 and 1
        virtual bool Initialize(T* flat_data_ptr, size_t m, size_t n, size_t l) = 0;
        // Reconstruction
        virtual bool PreReconstruct(size_t iter_num, size_t batch_size = 1) = 0;
        virtual bool Reconstruct(size_t epi, size_t iter_per_epi, size_t batch_size = 1) = 0;

}; // CpraInterface

} // name space CPRA

#include "./detail/CPRA_Common.inl"