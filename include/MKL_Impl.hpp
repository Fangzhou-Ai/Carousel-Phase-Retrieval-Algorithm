#pragma once
#ifdef HAS_MKL
#include "CPRA_Impl.hpp"
namespace CPRA
{
template <typename T>    
class MklImpl final : public CpraImpl<T>
{
    public:
        MklImpl(){}
        
        // Initialize
        bool Initialize(T* flat_data_ptr, size_t num) override;

        bool Forward2D(T* flat_input, T* flat_output, size_t m, size_t n, size_t batch_size, bool inplace = false) override{};

        bool Backward2D(T* flat_input, T* flat_output, size_t m, size_t n, size_t batch_size, bool inplace = false) override{};

        bool Forward3D(T* flat_input, T* flat_output, size_t m, size_t n, size_t l, size_t batch_size, bool inplace = false) override{};

        bool Backward3D(T* flat_input, T* flat_output, size_t m, size_t n, size_t l, size_t batch_size, bool inplace = false) override{};

        bool SpaceConstraint(T* flat_src_data, bool* flat_constr_data, size_t num, size_t batch_size) override;

        bool RealDataConstraint(std::complex<T>* flat_src_data, T* flat_constr_data, size_t num, size_t batch_size) override;

        bool ComplexDataConstraint(std::complex<T>* flat_src_data, std::complex<T>* flat_constr_data, size_t num, size_t batch_size) override;

        bool MergeAddData(T* flat_src, T* flat_dst, T alpha, size_t num) override;

        bool Normalization(T* flat_src, T norm, size_t num) override;

        bool Real2DTo3DInterpolation(T* flat_2d_src, T* flat_3D_dst, T* angles, size_t m, size_t n, size_t p, size_t l) override{};

        bool Complex2DTo3DInterpolation(std::complex<T>* flat_2d_src, std::complex<T>* flat_3D_dst, T* angles, size_t m, size_t n, size_t p, size_t l) override{};
    
#ifdef HAS_CUDA
        // CUDA version with cuda stream here
        bool Forward2D(T* flat_input, T* flat_output, size_t m, size_t n, size_t batch_size, cudaStream_t stream = 0, bool inplace = false) override {};

        bool Backward2D(T* flat_input, T* flat_output, size_t m, size_t n, size_t batch_size, cudaStream_t stream = 0, bool inplace = false) override {};

        bool Forward3D(T* flat_input, T* flat_output, size_t m, size_t n, size_t l, size_t batch_size, cudaStream_t stream = 0, bool inplace = false) override {};

        bool Backward3D(T* flat_input, T* flat_output, size_t m, size_t n, size_t l, size_t batch_size, cudaStream_t stream = 0, bool inplace = false) override {};

        bool SpaceConstraint(T* flat_src_data, bool* flat_constr_data, size_t num, size_t batch_size, cudaStream_t stream = 0) override {};

        bool RealDataConstraint(std::complex<T>* flat_src_data, T* flat_constr_data, size_t num, size_t batch_size, cudaStream_t stream = 0) override {};

        bool ComplexDataConstraint(std::complex<T>* flat_src_data, std::complex<T>* flat_constr_data, size_t num, size_t batch_size, cudaStream_t stream = 0) override {};

        // Add src to dst
        // flat_dst = alpha * flat_src + flat_dst 
        bool MergeAddData(T* flat_src, T* flat_dst, T alpha, size_t num, cudaStream_t stream = 0)override {};

        // flat_src = flat_src ./ norm
        bool Normalization(T* flat_src, T norm, size_t num, cudaStream_t stream = 0)override {};

#endif

        ~MklImpl(){}
};  // MklCpra

template<typename T>
std::unique_ptr<CpraImpl<T>> NewMKLImpl()
{
    return std::make_unique<MklImpl<T>>();
}

}// namespace CPRA

#include "./detail/MKL_Impl.inl"

#endif