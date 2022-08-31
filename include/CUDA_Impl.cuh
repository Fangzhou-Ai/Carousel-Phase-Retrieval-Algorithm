#pragma once
#ifdef HAS_CUDA
#include "CPRA_Impl.hpp"
#include "CUDA_Error_Check.hpp"

namespace CPRA
{
template <typename T>
class CudaImpl final : public CpraImpl<T>
{
    public:
        CudaImpl() = default;
        CudaImpl(size_t m, size_t n, size_t l, size_t batch_size){}

        // Initialize
        bool Initialize(T* flat_data_ptr, size_t num) override;

        // CUDA version with cuda stream here
        bool Forward2D(thrust::complex<T>* flat_input, cudaStream_t stream) override;

        bool Backward2D(thrust::complex<T>* flat_input, cudaStream_t stream) override {};

        bool Forward3D(thrust::complex<T>* flat_input, cudaStream_t stream) override {};

        bool Backward3D(thrust::complex<T>* flat_input, cudaStream_t stream) override {};

        bool SpaceConstraint(thrust::complex<T>* flat_src_data, T* flat_constr_data, size_t num, size_t batch_size, cudaStream_t stream) override {};

        bool RealDataConstraint(thrust::complex<T>* flat_src_data, T* flat_constr_data, size_t num, size_t batch_size, cudaStream_t stream) override {};

        bool ComplexDataConstraint(thrust::complex<T>* flat_src_data, thrust::complex<T>* flat_constr_data, size_t num, size_t batch_size, cudaStream_t stream) override {};

        // Add src to dst
        // flat_dst = alpha * flat_src + flat_dst 
        bool MergeAddData(thrust::complex<T>* flat_src, thrust::complex<T>* flat_dst, T alpha, T beta, size_t num, cudaStream_t stream) override {};

        // flat_src = flat_src ./ norm
        bool Normalization(thrust::complex<T>* flat_src, T norm, size_t num, cudaStream_t stream) override {};
        // Only support one rotating angle for now
        // param:
        // p : number of 2D sources
        // m, n, l: 3 dimensions
        // To interpolate real value, cast it to complex first
        bool Complex2DTo3DInterpolation(thrust::complex<T>* flat_2d_src, thrust::complex<T>* flat_3D_dst, T* angles, size_t m, size_t n, size_t p, size_t l, cudaStream_t stream) override {};
        
#ifdef HAS_MKL
        // MKL version, without cuda stream here
        bool Forward2D(std::complex<T>* flat_input) override {};

        bool Backward2D(std::complex<T>* flat_input) override {};

        bool Forward3D(std::complex<T>* flat_input) override {};

        bool Backward3D(std::complex<T>* flat_input) override {};

        bool SpaceConstraint(std::complex<T>* flat_src_data, T* flat_constr_data, size_t num, size_t batch_size) override {};

        bool RealDataConstraint(std::complex<T>* flat_src_data, T* flat_constr_data, size_t num, size_t batch_size) override {};

        bool ComplexDataConstraint(std::complex<T>* flat_src_data, std::complex<T>* flat_constr_data, size_t num, size_t batch_size) override {};

        // Add src to dst
        // flat_dst = alpha * flat_src + flat_dst 
        bool MergeAddData(std::complex<T>* flat_src, std::complex<T>* flat_dst, T alpha, T beta, size_t num) override {};

        // flat_src = flat_src ./ norm
        bool Normalization(std::complex<T>* flat_src, T norm, size_t batch_size) override {};

        bool Complex2DTo3DInterpolation(std::complex<T>* flat_2d_src, std::complex<T>* flat_3D_dst, T* angles, size_t m, size_t n, size_t p, size_t l) override {};
#endif

        ~CudaImpl(){}
}; // CudaCpra

template<typename T>
std::unique_ptr<CpraImpl<T>> NewCUDAImpl(size_t m, size_t n, size_t l, size_t batch_size)
{
    return std::make_unique<CudaImpl<T>>(m, n, l, batch_size);
}

} // namespace CPRA

#include "./detail/CUDA_Impl.inl"

#endif // HAS_CUDA