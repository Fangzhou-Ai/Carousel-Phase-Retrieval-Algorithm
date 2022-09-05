#pragma once
#ifdef HAS_CUDA
#include "CPRA_Impl.hpp"
#include "CUDA_Error_Check.hpp"
#include <cufft.h>
#include <thrust/complex.h>
#include <curand.h>
namespace CPRA
{
template <typename T>
class CudaImpl final : public CpraImpl<T>
{
    public:
        CudaImpl() = default;
        CudaImpl(unsigned long long m, unsigned long long n, unsigned long long l, unsigned long long batch_size)
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
            // Random Uniform
            CPRA_CURAND_CALL(curandCreateGenerator(&cuRandHandle, CURAND_RNG_PSEUDO_MT19937));
            CPRA_CURAND_CALL(curandSetPseudoRandomGeneratorSeed(cuRandHandle, time(NULL)));
        }

        // Initialize
        bool Initialize(T* flat_data_ptr, unsigned long long num) override;

        // CUDA version with cuda stream here
        bool Forward2D(std::complex<T>* flat_input) override;

        bool Backward2D(std::complex<T>* flat_input) override;

        bool Forward3D(std::complex<T>* flat_input) override;

        bool Backward3D(std::complex<T>* flat_input) override;

        bool SpaceConstraint(std::complex<T>* flat_src_data, T* flat_constr_data, unsigned long long num, unsigned long long batch_size) override;

        bool RealDataConstraint(std::complex<T>* flat_src_data, T* flat_constr_data, unsigned long long num, unsigned long long batch_size) override;

        bool ComplexDataConstraint(std::complex<T>* flat_src_data, std::complex<T>* flat_constr_data, unsigned long long num, unsigned long long batch_size) override;

        // Add src to dst
        // flat_dst = alpha * flat_src + flat_dst 
        bool MergeAddData(std::complex<T>* flat_src, std::complex<T>* flat_dst, T alpha, T beta, unsigned long long num) override;

        // flat_src = flat_src ./ norm
        bool Normalization(std::complex<T>* flat_src, T norm, unsigned long long num) override;
        // Only support one rotating angle for now
        // param:
        // p : number of 2D sources
        // m, n, l: 3 dimensions
        // To interpolate real value, cast it to complex first
        bool Complex2DTo3DInterpolation(std::complex<T>* flat_2d_src, std::complex<T>* flat_3D_dst, T* angles, unsigned long long m, unsigned long long n, unsigned long long p, unsigned long long l) override {};

        bool Memcpy(void* dst, void* src, unsigned long long bytes) override
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
        } 
    private:
        cudaStream_t stream_;
        cufftHandle Dfti2DHandle_;
        cufftHandle Dfti3DHandle_;
        curandGenerator_t cuRandHandle;
       
}; // CudaCpra

template<typename T>
std::unique_ptr<CpraImpl<T>> NewCUDAImpl(unsigned long long m, unsigned long long n, unsigned long long l, unsigned long long batch_size)
{
    return std::make_unique<CudaImpl<T>>(m, n, l, batch_size);
}

} // namespace CPRA

#include "./detail/CUDA_Impl.inl"

#endif // HAS_CUDA