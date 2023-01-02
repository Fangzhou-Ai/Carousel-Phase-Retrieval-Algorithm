#include <gtest/gtest.h>
#include "../include/CPRA.hpp"

TEST(MKLTEST, Test_MKL_Initialize)
{
    std::unique_ptr<CPRA::CpraImpl<float>> obj = CPRA::NewCpraImpl<float>(CPRA::IMPL_TYPE::MKL, 1, 1, 1, 1);
    float* test_ptr = (float*)mkl_malloc(sizeof(float) * 1000, 64);
    obj->Initialize(test_ptr, 1000);
    for(int i = 0; i < 1000; i++)
        EXPECT_TRUE((test_ptr[i] >= 0) && (test_ptr[i] <= 1));
    mkl_free(test_ptr);
}


TEST(MKLTEST, Test_Norm_Merge)
{
    std::unique_ptr<CPRA::CpraImpl<float>> obj = CPRA::NewCpraImpl<float>(CPRA::IMPL_TYPE::MKL, 1, 1, 1, 1);
    std::complex<float>* input = (std::complex<float>*)mkl_malloc(sizeof(std::complex<float>) * 1000, 64);
    for(auto i = 0; i < 1000; i++)
    {
        input[i].real(10);
        input[i].imag(10);
    }
    obj->MergeAddData(input, input, 0.5, 0.5, 1000);
    for(auto i = 0; i < 1000; i++)
    {
        EXPECT_EQ(input[i].real(), 10); 
        EXPECT_EQ(input[i].imag(), 10);   
    }
    
    obj->Normalization(input, 10, 1000);
    for(auto i = 0; i < 1000; i++)
    {
        EXPECT_EQ(input[i].real(), 1); 
        EXPECT_EQ(input[i].imag(), 1);   
    }
    

   mkl_free(input);
}

TEST(MKLTEST, Test_2D_DFT)
{
    std::unique_ptr<CPRA::CpraImpl<double>> obj = CPRA::NewCpraImpl<double>(CPRA::IMPL_TYPE::MKL, 10, 10, 1, 1);
    std::complex<double>* input = (std::complex<double>*)mkl_malloc(sizeof(std::complex<double  >) * 100, 64);
    for(auto i = 0; i < 100; i++)
    {
        input[i].real(1);
        input[i].imag(0);
    }
    obj->Forward2D(input);
    for(int i = 0; i < 10; i++)
    {
        for(int j = 0; j < 10; j++)
        {
            if(i == 0 && j == 0)EXPECT_NEAR(input[i * 10 + j].real(), 100, 1e-14);
            else EXPECT_NEAR(input[i * 10 + j].real(), 0, 1e-14);
            EXPECT_NEAR(input[i * 10 + j].imag(), 0, 1e-14);
        }
    }

    obj->Backward2D(input);
    for(int i = 0; i < 10; i++)
    {
        for(int j = 0; j < 10; j++)
        {
            EXPECT_NEAR(input[i * 10 + j].real(), 1, 1e-14);
            EXPECT_NEAR(input[i * 10 + j].imag(), 0, 1e-14);
        }
    }
    mkl_free(input);
}