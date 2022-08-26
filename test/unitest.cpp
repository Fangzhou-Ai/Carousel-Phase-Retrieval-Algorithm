#include <gtest/gtest.h>
#ifdef HAS_CUDA
#include "../include/Impl_Factory.hpp"
#include "../include/CPRA.hpp"

TEST(CUDATEST, Test_CUDA_Initialize)
{
    std::unique_ptr<CPRA::CpraImpl<float>> obj = CPRA::NewCpraImpl<float>(CPRA::IMPL_TYPE::CUDA);
    float* test_ptr;
    cudaMallocManaged((void**) & test_ptr, sizeof(float) * 1000);
    obj->Initialize(test_ptr, 1000);
    for(int i = 0; i < 1000; i++)
        EXPECT_TRUE((test_ptr[i] >= 0) && (test_ptr[i] <= 1));
    cudaFree(test_ptr);
}
#endif

#ifdef HAS_MKL
#include "../include/MKL_Impl.hpp"
TEST(MKLTEST, Test_MKL_Initialize)
{
    std::unique_ptr<CPRA::CpraImpl<float>> obj = CPRA::NewCpraImpl<float>(CPRA::IMPL_TYPE::MKL);
    float* test_ptr = (float*)mkl_malloc(sizeof(float) * 1000, 64);
    obj->Initialize(test_ptr, 1000);
    for(int i = 0; i < 1000; i++)
        EXPECT_TRUE((test_ptr[i] >= 0) && (test_ptr[i] <= 1));
    mkl_free(test_ptr);
}
#endif



TEST(CPRATEST, Test_IO_Host_BINARY)
{
    CPRA::Cpra<float, CPRA::IMPL_TYPE::CUDA> obj;
    float* output_ptr;
    cudaMallocManaged((void**) & output_ptr, sizeof(float) * 1000);
    for(int i = 0; i < 1000; i++)
        output_ptr[i] = i;
    EXPECT_EQ(obj.WriteMatrixToFile("/tmp/test_output.bin", output_ptr, 10, 10, 10), true);

    float* Input_ptr;
    cudaMallocManaged((void**) & Input_ptr, sizeof(float) * 1000);
    EXPECT_EQ(obj.ReadMatrixFromFile("/tmp/test_output.bin", Input_ptr, 10, 10, 10), true);
    for(int i = 0; i < 1000; i++)
        EXPECT_EQ(Input_ptr[i], i);

    cudaFree(output_ptr);
    cudaFree(Input_ptr);
}

