#include <iostream>
#include <sstream>
#include <chrono> 
#include "../include/CPRA.hpp"


static constexpr uint64_t M  = 512;
static constexpr uint64_t N = 512;
static constexpr uint64_t P = 256;
static constexpr uint64_t L = 512;
static constexpr uint64_t BATCHSIZE_CPRA = 8;
static constexpr uint64_t BATCHSIZE_CONV = 8;
static constexpr float BETA = 0.9;
void ShrinkWrap_CPRA_MKL_Sample(int epi, int iter)
{
    // 2D part test
    // Shrinkwrap
    CPRA::Cpra<float, CPRA::IMPL_TYPE::MKL> obj1(M, N, L, BATCHSIZE_CPRA);
    //CPRA::Cpra<float, CPRA::IMPL_TYPE::MKL> obj2(M, N, L, BATCHSIZE_CPRA);
    std::complex<float>* random_guess = (std::complex<float>*)obj1.allocate(sizeof(std::complex<float>) * M * N * P * BATCHSIZE_CPRA);
    std::complex<float>* t_random_guess_1 = (std::complex<float>*)obj1.allocate(sizeof(std::complex<float>) * M * N * BATCHSIZE_CPRA);
    std::complex<float>* t_random_guess_2 = (std::complex<float>*)obj1.allocate(sizeof(std::complex<float>) * M * N * BATCHSIZE_CPRA);
    obj1.impl_->Initialize(reinterpret_cast<float*>(random_guess), M * N * P * BATCHSIZE_CPRA * 2);
    obj1.impl_->Initialize(reinterpret_cast<float*>(t_random_guess_1), M * N * BATCHSIZE_CPRA * 2);
    obj1.impl_->Initialize(reinterpret_cast<float*>(t_random_guess_2), M * N * BATCHSIZE_CPRA * 2);


    float* dataConstr = (float*)obj1.allocate(sizeof(float) * M * N * P);
    obj1.impl_->Initialize(dataConstr, M * N * P);
    float* spaceConstr = (float*)obj1.allocate(sizeof(float) * M * N * P);
    obj1.impl_->Initialize(spaceConstr, M * N * P);


    
    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();

    for(auto e = 0; e < epi; e++)  // episode
    {
        for(auto p = 0; p < P; p++) // each projected object
        {


            // Merge data
            obj1.impl_->MergeAddData(random_guess + M * N * BATCHSIZE_CPRA * ((p + 1) % P), random_guess + M * N * BATCHSIZE_CPRA * p, 0.5, 1.0, M * N * BATCHSIZE_CPRA);
            obj1.impl_->MergeAddData(random_guess + M * N * BATCHSIZE_CPRA * ((p - 1 + P) % P), random_guess + M * N * BATCHSIZE_CPRA * p, 0.5, 1.0, M * N * BATCHSIZE_CPRA);
            obj1.impl_->Normalization(random_guess + M * N * BATCHSIZE_CPRA * p, 2, M * N * BATCHSIZE_CPRA);


            // Copy to temporary variable
            obj1.impl_->Memcpy((void*)(t_random_guess_1), (void*)(random_guess + M * N * BATCHSIZE_CPRA * p), sizeof(std::complex<float>) * M * N * BATCHSIZE_CPRA);
            obj1.impl_->Memcpy((void*)(t_random_guess_2), (void*)(random_guess + M * N * BATCHSIZE_CPRA * p), sizeof(std::complex<float>) * M * N * BATCHSIZE_CPRA);
            // Reconstruct
            for(auto i = 0; i < iter; i++) // each iteration
            {
                // Part 1
                obj1.impl_->Forward2D(t_random_guess_1);
                obj1.impl_->RealDataConstraint(t_random_guess_1, dataConstr + M * N * p, M * N * BATCHSIZE_CPRA, BATCHSIZE_CPRA);
                obj1.impl_->Backward2D(t_random_guess_1);
                // Sync here to make sure correct result
                //obj1.impl_->Sync();
                obj1.impl_->MergeAddData(t_random_guess_1, random_guess + M * N * BATCHSIZE_CPRA * p, 1.0 + 1.0 / BETA, -1.0 / BETA, M * N * BATCHSIZE_CPRA);
                obj1.impl_->SpaceConstraint(t_random_guess_1, spaceConstr + M * N * p, M * N * BATCHSIZE_CPRA, BATCHSIZE_CPRA);
                
                // Part 2
                obj1.impl_->SpaceConstraint(t_random_guess_2, spaceConstr + M * N * p, M * N * BATCHSIZE_CPRA, BATCHSIZE_CPRA);
                // Sync here to make sure correct result
                //obj1.impl_->Sync();
                obj1.impl_->MergeAddData(t_random_guess_2, random_guess + M * N * BATCHSIZE_CPRA * p, 1.0 - 1.0 / BETA, -1.0 / BETA, M * N * BATCHSIZE_CPRA);
                obj1.impl_->Forward2D(t_random_guess_2);
                obj1.impl_->RealDataConstraint(t_random_guess_2, dataConstr + M * N * p, M * N * BATCHSIZE_CPRA, BATCHSIZE_CPRA);
                obj1.impl_->Backward2D(t_random_guess_2);

                // Merge 
                obj1.impl_->MergeAddData(t_random_guess_1, random_guess + M * N * BATCHSIZE_CPRA * p, BETA, 1.0, M * N * BATCHSIZE_CPRA);
                //obj1.impl_->Sync();
                obj1.impl_->MergeAddData(t_random_guess_2, random_guess + M * N * BATCHSIZE_CPRA * p, -1.0 * BETA, 1.0, M * N * BATCHSIZE_CPRA);
                //obj1.impl_->Sync();
            }
            
        }
    }
    obj1.impl_->Sync();
    //obj2.impl_->Sync();
    end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    std::cout << "Finished computation at " << std::ctime(&end_time)
              << "elapsed time: " << elapsed_seconds.count() << "s\n";

    obj1.deallocate(random_guess);
    obj1.deallocate(t_random_guess_1);
    obj1.deallocate(t_random_guess_2);
    obj1.deallocate(dataConstr);
    obj1.deallocate(spaceConstr);
    return;
}

void ShrinkWrap_Conventional_MKL_Sample(int iter)
{
    CPRA::Cpra<float, CPRA::IMPL_TYPE::MKL> obj1(M, N, L, BATCHSIZE_CONV);
    std::complex<float>* random_guess = (std::complex<float>*)obj1.allocate(sizeof(std::complex<float>) * M * N * L * BATCHSIZE_CONV);
    std::complex<float>* t_random_guess_1 = (std::complex<float>*)obj1.allocate(sizeof(std::complex<float>) * M * N * L * BATCHSIZE_CONV);
    std::complex<float>* t_random_guess_2 = (std::complex<float>*)obj1.allocate(sizeof(std::complex<float>) * M * N * L * BATCHSIZE_CONV);
    obj1.impl_->Initialize(reinterpret_cast<float*>(random_guess), M * N * L * BATCHSIZE_CONV * 2);
    obj1.impl_->Initialize(reinterpret_cast<float*>(t_random_guess_1), M * N * L * BATCHSIZE_CONV * 2);
    obj1.impl_->Initialize(reinterpret_cast<float*>(t_random_guess_2), M * N * L * BATCHSIZE_CONV * 2);


    float* dataConstr = (float*)obj1.allocate(sizeof(float) * M * N * L);
    obj1.impl_->Initialize(dataConstr, M * N * L);
    float* spaceConstr = (float*)obj1.allocate(sizeof(float) * M * N * L);
    obj1.impl_->Initialize(spaceConstr, M * N * L);


    
    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();

    // Copy to temporary variable
    obj1.impl_->Memcpy((void*)(t_random_guess_1), (void*)(random_guess), sizeof(std::complex<float>) * M * N * L * BATCHSIZE_CONV);
    obj1.impl_->Memcpy((void*)(t_random_guess_2), (void*)(random_guess), sizeof(std::complex<float>) * M * N * L * BATCHSIZE_CONV);
    // Reconstruct
    for(auto i = 0; i < iter; i++) // each iteration
    {
        // Part 1
        obj1.impl_->Forward2D(t_random_guess_1);
        obj1.impl_->RealDataConstraint(t_random_guess_1, dataConstr, M * N * L * BATCHSIZE_CONV, BATCHSIZE_CONV);
        obj1.impl_->Backward2D(t_random_guess_1);
        obj1.impl_->MergeAddData(t_random_guess_1, random_guess, 1.0 + 1.0 / BETA, -1.0 / BETA, M * N * L * BATCHSIZE_CONV);
        obj1.impl_->SpaceConstraint(t_random_guess_1, spaceConstr, M * N * L * BATCHSIZE_CONV, BATCHSIZE_CONV);
        
        // Part 2
        obj1.impl_->SpaceConstraint(t_random_guess_2, spaceConstr, M * N * L * BATCHSIZE_CONV, BATCHSIZE_CONV);
        obj1.impl_->MergeAddData(t_random_guess_2, random_guess, 1.0 - 1.0 / BETA, -1.0 / BETA, M * N * L * BATCHSIZE_CONV);
        obj1.impl_->Forward2D(t_random_guess_2);
        obj1.impl_->RealDataConstraint(t_random_guess_2, dataConstr, M * N * L * BATCHSIZE_CONV, BATCHSIZE_CONV);
        obj1.impl_->Backward2D(t_random_guess_2);

        // Merge 
        obj1.impl_->MergeAddData(t_random_guess_1, random_guess, BETA, 1.0, M * N * L * BATCHSIZE_CONV);
        obj1.impl_->MergeAddData(t_random_guess_2, random_guess, -1.0 * BETA, 1.0, M * N * L * BATCHSIZE_CONV);
    }
    obj1.impl_->Sync();
    //obj2.impl_->Sync();
    end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    std::cout << "Finished computation at " << std::ctime(&end_time)
              << "elapsed time: " << elapsed_seconds.count() << "s\n";

    obj1.deallocate(random_guess);
    obj1.deallocate(t_random_guess_1);
    obj1.deallocate(t_random_guess_2);
    obj1.deallocate(dataConstr);
    obj1.deallocate(spaceConstr);
    return;
}


int main(int argc, char *argv[])
{
    if(argc == 3)
    {
        std::istringstream epi_ss(argv[1]);
        std::istringstream iter_ss(argv[2]);
        int epi;
        int iter;
        epi_ss >> epi;
        iter_ss >> iter;
        std::cout << "Running CPRA benchmark, epi is " << epi << " iter is " << iter << std::endl;
        ShrinkWrap_CPRA_MKL_Sample(epi, iter);
    }
    else if (argc == 2)
    {
        std::istringstream iter_ss(argv[1]);
        int iter;
        iter_ss >> iter;
        std::cout << "Running Conventional benchmark, iter is " << iter << std::endl;
        ShrinkWrap_Conventional_MKL_Sample(iter);
    }
    else
    {
        std::cerr << "Wrong num of command line arg!" << std::endl;
    }
    return 0;
}