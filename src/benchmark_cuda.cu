#include <iostream>
#include <sstream>
#include <chrono> 
#include "../include/CPRA.hpp"


#define M 4096
#define N 4096
#define P 32
#define L 4096
#define BATCHSIZE 1
#define BETA 0.9
void ShrinkWrap_CUDA_Sample(int epi, int iter)
{
    // 2D part test
    // Shrinkwrap
    CPRA::Cpra<float, CPRA::IMPL_TYPE::CUDA> obj1(M, N, L, BATCHSIZE);
    //CPRA::Cpra<float, CPRA::IMPL_TYPE::CUDA> obj1(M, N, L, BATCHSIZE);
    std::complex<float>* random_guess = (std::complex<float>*)obj1.allocate(sizeof(std::complex<float>) * M * N * P * BATCHSIZE);
    std::complex<float>* t_random_guess_1 = (std::complex<float>*)obj1.allocate(sizeof(std::complex<float>) * M * N * BATCHSIZE);
    std::complex<float>* t_random_guess_2 = (std::complex<float>*)obj1.allocate(sizeof(std::complex<float>) * M * N * BATCHSIZE);
    obj1.impl_->Initialize((float*)random_guess, M * N * P * BATCHSIZE * 2);
    obj1.impl_->Initialize((float*)t_random_guess_1, M * N * BATCHSIZE * 2);
    obj1.impl_->Initialize((float*)t_random_guess_2, M * N * BATCHSIZE * 2);


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
            obj1.impl_->MergeAddData(random_guess + M * N * BATCHSIZE * ((p + 1) % P), random_guess + M * N * BATCHSIZE * p, 0.5, 1.0, M * N * BATCHSIZE);
            obj1.impl_->MergeAddData(random_guess + M * N * BATCHSIZE * ((p - 1 + P) % P), random_guess + M * N * BATCHSIZE * p, 0.5, 1.0, M * N * BATCHSIZE);
            obj1.impl_->Normalization(random_guess + M * N * BATCHSIZE * p, 2, M * N * BATCHSIZE);


            // Copy to temporary variable
            obj1.impl_->Memcpy((void*)(t_random_guess_1), (void*)(random_guess + M * N * BATCHSIZE * p), sizeof(std::complex<float>) * M * N * BATCHSIZE);
            obj1.impl_->Memcpy((void*)(t_random_guess_2), (void*)(random_guess + M * N * BATCHSIZE * p), sizeof(std::complex<float>) * M * N * BATCHSIZE);
            // Reconstruct
            for(auto i = 0; i < iter; i++) // each iteration
            {
                // Part 1
                obj1.impl_->Forward2D(t_random_guess_1);
                obj1.impl_->RealDataConstraint(t_random_guess_1, dataConstr + M * N * p, M * N * BATCHSIZE, BATCHSIZE);
                obj1.impl_->Backward2D(t_random_guess_1);
                // Sync here to make sure correct result
                //obj1.impl_->Sync();
                obj1.impl_->MergeAddData(t_random_guess_1, random_guess + M * N * BATCHSIZE * p, 1.0 + 1.0 / BETA, -1.0 / BETA, M * N * BATCHSIZE);
                obj1.impl_->SpaceConstraint(t_random_guess_1, spaceConstr + M * N * p, M * N * BATCHSIZE, BATCHSIZE);
                
                // Part 2
                obj1.impl_->SpaceConstraint(t_random_guess_2, spaceConstr + M * N * p, M * N * BATCHSIZE, BATCHSIZE);
                // Sync here to make sure correct result
                //obj1.impl_->Sync();
                obj1.impl_->MergeAddData(t_random_guess_2, random_guess + M * N * BATCHSIZE * p, 1.0 - 1.0 / BETA, -1.0 / BETA, M * N * BATCHSIZE);
                obj1.impl_->Forward2D(t_random_guess_2);
                obj1.impl_->RealDataConstraint(t_random_guess_2, dataConstr + M * N * p, M * N * BATCHSIZE, BATCHSIZE);
                obj1.impl_->Backward2D(t_random_guess_2);

                // Merge 
                obj1.impl_->MergeAddData(t_random_guess_1, random_guess + M * N * BATCHSIZE * p, BETA, 1.0, M * N * BATCHSIZE);
                //obj1.impl_->Sync();
                obj1.impl_->MergeAddData(t_random_guess_2, random_guess + M * N * BATCHSIZE * p, -1.0 * BETA, 1.0, M * N * BATCHSIZE);
                //obj1.impl_->Sync();
            }
            
        }
    }
    CPRA_CUDA_TRY(cudaDeviceSynchronize());
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
}


int main(int argc, char *argv[])
{
    if(argc != 3)
    {
        std::cerr << "Wrong num of command line arg!" << std::endl;
    }
    std::istringstream epi_ss(argv[1]);
    std::istringstream iter_ss(argv[2]);
    int epi;
    int iter;
    epi_ss >> epi;
    iter_ss >> iter;
    std::cout << "epi is " << epi << " iter is " << iter << std::endl;
    ShrinkWrap_CUDA_Sample(epi, iter);
}