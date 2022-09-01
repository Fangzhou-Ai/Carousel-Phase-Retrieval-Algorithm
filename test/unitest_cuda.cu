#include <gtest/gtest.h>
#include "../include/CPRA.hpp"


TEST(CUDATEST, init)
{
    std::unique_ptr<CPRA::CpraImpl<float>> obj = CPRA::NewCpraImpl<float>(CPRA::IMPL_TYPE::CUDA, 1, 1, 1, 1);
}