#include <gtest/gtest.h>
#ifdef HAS_CUDA
#include "../include/CUDA_CPRA.cuh"

TEST(CUDATEST, Test_Initialize)
{
    CPRA::CudaCpra<float> obj;
    auto test_ptr = obj.Initialize(10, 10, 10);
    for(int i = 0; i < 1000; i++)
        EXPECT_TRUE((test_ptr[i] >= 0) && (test_ptr[i] <= 1));
}

TEST(CUDATEST, Test_IO_Host_BINARY)
{
    CPRA::CudaCpra<float> obj;
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
}

#endif