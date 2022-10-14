#pragma once
#ifdef HAS_CUDA
#include "CPRA_Impl.hpp"
#include "CUDA_Error_Check.hpp"
#include <cufft.h>
#include <thrust/complex.h>
#include <curand.h>
#include <cublas_v2.h>
namespace CPRA
{
template <typename T>
class CudaImpl final : public CpraImpl<T>
{
    public:
        CudaImpl() = default;
        CudaImpl(uint64_t m, uint64_t n, uint64_t l, uint64_t batch_size) : m_(m), n_(n), l_(l), batch_(batch_size)
        {
            int Dim2D[2] = {m, n};
            int Dim3D[3] = {m, n, l};
            if constexpr (std::is_same_v<T, float>)
            {
                cufftPlanMany(&Dfti2DHandle_, 2, Dim2D,
                    NULL, 1, 0,
                    NULL, 1, 0,
                    CUFFT_C2C, batch_size);
                cufftPlanMany(&Dfti3DHandle_, 3, Dim3D,
                    NULL, 1, 0,
                    NULL, 1, 0,
                    CUFFT_C2C, batch_size);
            }
            else
            {
                cufftPlanMany(&Dfti2DHandle_, 2, Dim2D,
                    NULL, 1, 0,
                    NULL, 1, 0,
                    CUFFT_Z2Z, batch_size);
                cufftPlanMany(&Dfti3DHandle_, 3, Dim3D,
                    NULL, 1, 0,
                    NULL, 1, 0,
                    CUFFT_Z2Z, batch_size);
            }
            CPRA_CUDA_TRY(cudaStreamCreate(&stream_));
            cufftSetStream(Dfti2DHandle_, stream_);
            cufftSetStream(Dfti3DHandle_, stream_);
            // Random Uniform, not thread safe. read this
            // https://docs.nvidia.com/cuda/curand/host-api-overview.html#thread-safety
            CPRA_CURAND_CALL(curandCreateGenerator(&cuRandHandle, CURAND_RNG_PSEUDO_MT19937));
            CPRA_CURAND_CALL(curandSetPseudoRandomGeneratorSeed(cuRandHandle, time(NULL)));
            // cuBLAS
            CPRA_CUBLAS_CALL(cublasCreate(&cuBlasHandle));
            CPRA_CUBLAS_CALL(cublasSetStream(cuBlasHandle, stream_));
        }

        // Initialize
        bool Initialize(T* flat_data_ptr, uint64_t num) override;

        // CUDA version with cuda stream here
        bool Forward2D(std::complex<T>* flat_input) override;

        bool Backward2D(std::complex<T>* flat_input) override;

        bool FFTShift2D(std::complex<T>* flat_input, size_t M, size_t N, size_t Batch) override;

        bool Forward3D(std::complex<T>* flat_input) override;

        bool Backward3D(std::complex<T>* flat_input) override;

        bool FFTShift3D(std::complex<T>* flat_input, size_t M, size_t N, size_t L, size_t Batch) override;

        bool SpaceConstraint(std::complex<T>* flat_src_data, T* flat_constr_data, uint64_t num, uint64_t batch_size) override;

        bool DataConstraint(std::complex<T>* flat_src_data, T* flat_constr_data, uint64_t num, uint64_t batch_size) override;

        bool DataConstraint(std::complex<T>* flat_src_data, std::complex<T>* flat_constr_data, uint64_t num, uint64_t batch_size) override;

        // Add src to dst
        // flat_dst = alpha * flat_src + flat_dst 
        bool MergeAddData(std::complex<T>* flat_src, std::complex<T>* flat_dst, T alpha, T beta, uint64_t num) override;

        // flat_src = flat_src ./ norm
        bool Normalization(std::complex<T>* flat_src, T norm, uint64_t num) override;
        // Only support one rotating angle for now
        // param:
        // p : number of 2D sources
        // m, n, l: 3 dimensions
        
        bool Real2DTo3DInterpolation(T* flat_2d_src, T* flat_3D_dst, T* angles, uint64_t m, uint64_t n, uint64_t p, uint64_t l) override;

        bool Complex2DTo3DInterpolation(std::complex<T>* flat_2d_src, std::complex<T>* flat_3D_dst, T* angles, uint64_t m, uint64_t n, uint64_t p, uint64_t l) override {return true;}

        bool ConvergeError(std::complex<T>* flat_old, std::complex<T>* flat_new, T* flat_error, uint64_t m, uint64_t n, uint64_t l = 1, uint64_t batch_size = 1) override;

        bool Memcpy(void* dst, void* src, uint64_t bytes) override
        {
            CPRA_CUDA_TRY(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDefault, stream_));
            return true;
        }

        bool Sync() override
        {
            CPRA_CUDA_TRY(cudaStreamSynchronize(stream_));
            return true;
        }

        ~CudaImpl()
        {
            cufftDestroy(Dfti2DHandle_);
            cufftDestroy(Dfti3DHandle_);
            CPRA_CUDA_TRY(cudaStreamDestroy(stream_));
            CPRA_CURAND_CALL(curandDestroyGenerator(cuRandHandle));
            CPRA_CUBLAS_CALL(cublasDestroy(cuBlasHandle));
        } 
    private:
        cudaStream_t stream_;
        cufftHandle Dfti2DHandle_;
        cufftHandle Dfti3DHandle_;
        cublasHandle_t cuBlasHandle;
        curandGenerator_t cuRandHandle;
        const uint64_t m_, n_, l_, batch_;
       
}; // CudaCpra

template<typename T>
std::unique_ptr<CpraImpl<T>> NewCUDAImpl(uint64_t m, uint64_t n, uint64_t l, uint64_t batch_size)
{
    return std::make_unique<CudaImpl<T>>(m, n, l, batch_size);
}

} // namespace CPRA

#include "./detail/CUDA_Impl.inl"

#endif // HAS_CUDA