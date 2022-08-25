#include <fstream>
#include <cstdio>

namespace CPRA{

template<typename T>
bool CpraInterface<T>::ReadMatrixFromFile(std::string FileName, T* flat_data_ptr, size_t m, size_t n, size_t l)
{
    if(!std::filesystem::exists(FileName))
    {
        throw std::runtime_error("Reading input file " + FileName + " not exist!");
        return false;
    }

    if(m == 0 || n == 0 || l == 0)
    {
        throw std::runtime_error("Wrong reading size, 0 dimention detected!");
        return false;
    }


    FILE* inputfile = fopen(FileName.c_str(), "rb");
    fread(flat_data_ptr, sizeof(T), m * n * l, inputfile);
    fclose(inputfile);
    return true;
}

template<typename T>
bool CpraInterface<T>::WriteMatrixToFile(std::string FileName, T* flat_data_ptr, size_t m, size_t n, size_t l)
{
    if(std::filesystem::exists(FileName))
    {
        std::cout << "We're over writting file " << FileName << "!" << std::endl;
    }

    if(m == 0 || n == 0 || l == 0)
    {
        throw std::runtime_error("Wrong writing size, 0 dimention detected!");
        return false;
    }
    
    FILE *outputfile = fopen(FileName.c_str(), "wb");
    fwrite(flat_data_ptr, sizeof(T), m * n * l, outputfile);
    fclose(outputfile);
    return true;
}


}