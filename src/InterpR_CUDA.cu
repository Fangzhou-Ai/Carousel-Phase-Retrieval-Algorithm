#include "../include/CPRA.hpp"
#include "../include/Command_Parser.hpp"
#include <iostream>
#include <sstream>
#include <chrono> 
#include <vector>
#include <string>
#include <utility>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#define PI 3.1415926

int main(int argc, const char* argv[])
{
    CPRA::Parser parser;
    parser.parse(argc, argv);
    CPRA::Cpra<float, CPRA::IMPL_TYPE::CUDA> obj(1, 1, 1, 1);
    float* flat_2d_src = (float*)obj.allocate(sizeof(float) * parser.getM() * parser.getN() * parser.getP());
    if(!obj.ReadMatrixFromFile(parser.getDataConstr(), flat_2d_src, parser.getM(), parser.getN(), parser.getP()))
    {
        throw std::runtime_error("Read file " + parser.getDataConstr() + " failed!");
    }
    float* flat_3d_dst = (float*)obj.allocate(sizeof(float) * parser.getM() * parser.getN() * parser.getL());

    thrust::host_vector<float> h_angles(parser.getP());
    for(int i = 0; i < parser.getP(); i++)
    {
        h_angles[i] = i * PI / (parser.getP() - 1);
    }
    thrust::device_vector<float> angles = h_angles;
    auto angles_ptr = thrust::raw_pointer_cast(angles.data());
    obj.impl_->Real2DTo3DInterpolation(flat_2d_src, flat_3d_dst, angles_ptr, parser.getM(), parser.getN(), parser.getP(), parser.getL());
    obj.WriteMatrixToFile(parser.getOutputReconstr(), flat_3d_dst, parser.getM() * parser.getN() * parser.getL(), 1, 1);
    obj.deallocate(flat_2d_src);
    obj.deallocate(flat_3d_dst);
    return EXIT_SUCCESS;
}