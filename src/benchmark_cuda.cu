#include <iostream>
#include <sstream>
#include <chrono> 
#include "../include/CPRA.hpp"


static constexpr uint64_t M  = 256;
static constexpr uint64_t N = 256;
static constexpr uint64_t P = 128;
static constexpr uint64_t L = 256;
static constexpr uint64_t BATCHSIZE_CPRA = 16;
static constexpr uint64_t BATCHSIZE_CONV = 1;
static constexpr float BETA = 0.9;

void ShrinkWrap_CPRA_CUDA_Sample(int epi, int iter)
{
    // 2D part test
    // Shrinkwrap
    CPRA::Cpra<float, CPRA::IMPL_TYPE::CUDA> compute(M, N, L, BATCHSIZE_CPRA);
    //CPRA::Cpra<float, CPRA::IMPL_TYPE::CUDA> memory(1, 1, 1, 1);
    std::complex<float>* random_guess = (std::complex<float>*)compute.allocate(sizeof(std::complex<float>) * M * N * P * BATCHSIZE_CPRA);
    std::complex<float>* t_random_guess_1 = (std::complex<float>*)compute.allocate(sizeof(std::complex<float>) * M * N * BATCHSIZE_CPRA);
    std::complex<float>* t_random_guess_2 = (std::complex<float>*)compute.allocate(sizeof(std::complex<float>) * M * N * BATCHSIZE_CPRA);
    compute.impl_->Initialize(reinterpret_cast<float*>(random_guess), M * N * P * BATCHSIZE_CPRA * 2);
    compute.impl_->Initialize(reinterpret_cast<float*>(t_random_guess_1), M * N * BATCHSIZE_CPRA * 2);
    compute.impl_->Initialize(reinterpret_cast<float*>(t_random_guess_2), M * N * BATCHSIZE_CPRA * 2);


    float* dataConstr = (float*)compute.allocate(sizeof(float) * M * N * P);
    compute.impl_->Initialize(dataConstr, M * N * P);
    float* spaceConstr = (float*)compute.allocate(sizeof(float) * M * N * P);
    compute.impl_->Initialize(spaceConstr, M * N * P);

    std::chrono::time_point<std::chrono::system_clock> start, endA, endB, endC, endD;
    std::chrono::duration<double> elapsed_seconds;
    std::time_t end_time;

    start = std::chrono::system_clock::now();

    // Step A, pre-reconstruct
    // Num of iteration here is fixed at 1000
    // Shrinkwrap algo
    compute.impl_->Memcpy((void*)(t_random_guess_1), (void*)(random_guess), sizeof(std::complex<float>) * M * N * BATCHSIZE_CPRA);
    compute.impl_->Memcpy((void*)(t_random_guess_2), (void*)(random_guess), sizeof(std::complex<float>) * M * N * BATCHSIZE_CPRA);
    for(auto i = 0; i < 1000; i++)
    {
        // Part 1
        compute.impl_->Forward2D(t_random_guess_1);
        compute.impl_->RealDataConstraint(t_random_guess_1, dataConstr, M * N * BATCHSIZE_CPRA, BATCHSIZE_CPRA);
        compute.impl_->Backward2D(t_random_guess_1);
        // Sync here to make sure correct result
        //compute.impl_->Sync();
        compute.impl_->MergeAddData(t_random_guess_1, random_guess, 1.0 + 1.0 / BETA, -1.0 / BETA, M * N * BATCHSIZE_CPRA);
        compute.impl_->SpaceConstraint(t_random_guess_1, spaceConstr, M * N * BATCHSIZE_CPRA, BATCHSIZE_CPRA);
        
        // Part 2
        compute.impl_->SpaceConstraint(t_random_guess_2, spaceConstr, M * N * BATCHSIZE_CPRA, BATCHSIZE_CPRA);
        // Sync here to make sure correct result
        //compute.impl_->Sync();
        compute.impl_->MergeAddData(t_random_guess_2, random_guess, 1.0 - 1.0 / BETA, -1.0 / BETA, M * N * BATCHSIZE_CPRA);
        compute.impl_->Forward2D(t_random_guess_2);
        compute.impl_->RealDataConstraint(t_random_guess_2, dataConstr, M * N * BATCHSIZE_CPRA, BATCHSIZE_CPRA);
        compute.impl_->Backward2D(t_random_guess_2);

        // Merge 
        compute.impl_->MergeAddData(t_random_guess_1, random_guess, BETA, 1.0, M * N * BATCHSIZE_CPRA);
        //compute.impl_->Sync();
        compute.impl_->MergeAddData(t_random_guess_2, random_guess, -1.0 * BETA, 1.0, M * N * BATCHSIZE_CPRA);
        //compute.impl_->Sync();
    }
    compute.impl_->Sync();

    endA = std::chrono::system_clock::now();
    elapsed_seconds = endA - start;
    end_time = std::chrono::system_clock::to_time_t(endA);
    std::cout << "Finished step A computation at " << std::ctime(&end_time)
              << "elapsed time: " << elapsed_seconds.count() << "s\n";
    // Step B, reconstruct 2D projected object
    for(auto e = 0; e < epi; e++)  // episode
    {
        for(auto p = 0; p < P; p++) // each projected object
        {
           // Merge data
            compute.impl_->MergeAddData(random_guess + M * N * BATCHSIZE_CPRA * ((p + 1) % P), random_guess + M * N * BATCHSIZE_CPRA * p, 0.5, 1.0, M * N * BATCHSIZE_CPRA);
            compute.impl_->MergeAddData(random_guess + M * N * BATCHSIZE_CPRA * ((p - 1 + P) % P), random_guess + M * N * BATCHSIZE_CPRA * p, 0.5, 1.0, M * N * BATCHSIZE_CPRA);
            compute.impl_->Normalization(random_guess + M * N * BATCHSIZE_CPRA * p, 2, M * N * BATCHSIZE_CPRA);

            // Copy to temporary variable
            compute.impl_->Memcpy((void*)(t_random_guess_1), (void*)(random_guess + M * N * BATCHSIZE_CPRA * p), sizeof(std::complex<float>) * M * N * BATCHSIZE_CPRA);
            compute.impl_->Memcpy((void*)(t_random_guess_2), (void*)(random_guess + M * N * BATCHSIZE_CPRA * p), sizeof(std::complex<float>) * M * N * BATCHSIZE_CPRA);
            // Reconstruct
            // Shrinkwrap algo
            for(auto i = 0; i < iter; i++) // each iteration
            {
                // Part 1
                compute.impl_->Forward2D(t_random_guess_1);
                compute.impl_->RealDataConstraint(t_random_guess_1, dataConstr + M * N * p, M * N * BATCHSIZE_CPRA, BATCHSIZE_CPRA);
                compute.impl_->Backward2D(t_random_guess_1);
                // Sync here to make sure correct result
                //compute.impl_->Sync();
                compute.impl_->MergeAddData(t_random_guess_1, random_guess + M * N * BATCHSIZE_CPRA * p, 1.0 + 1.0 / BETA, -1.0 / BETA, M * N * BATCHSIZE_CPRA);
                compute.impl_->SpaceConstraint(t_random_guess_1, spaceConstr + M * N * p, M * N * BATCHSIZE_CPRA, BATCHSIZE_CPRA);
                
                // Part 2
                compute.impl_->SpaceConstraint(t_random_guess_2, spaceConstr + M * N * p, M * N * BATCHSIZE_CPRA, BATCHSIZE_CPRA);
                // Sync here to make sure correct result
                //compute.impl_->Sync();
                compute.impl_->MergeAddData(t_random_guess_2, random_guess + M * N * BATCHSIZE_CPRA * p, 1.0 - 1.0 / BETA, -1.0 / BETA, M * N * BATCHSIZE_CPRA);
                compute.impl_->Forward2D(t_random_guess_2);
                compute.impl_->RealDataConstraint(t_random_guess_2, dataConstr + M * N * p, M * N * BATCHSIZE_CPRA, BATCHSIZE_CPRA);
                compute.impl_->Backward2D(t_random_guess_2);

                // Merge 
                compute.impl_->MergeAddData(t_random_guess_1, random_guess + M * N * BATCHSIZE_CPRA * p, BETA, 1.0, M * N * BATCHSIZE_CPRA);
                //compute.impl_->Sync();
                compute.impl_->MergeAddData(t_random_guess_2, random_guess + M * N * BATCHSIZE_CPRA * p, -1.0 * BETA, 1.0, M * N * BATCHSIZE_CPRA);
                //compute.impl_->Sync();
            }
            
        }
    }
    compute.impl_->Sync();
    //memory.impl_->Sync();

    endB = std::chrono::system_clock::now();
    elapsed_seconds = endB - endA;
    end_time = std::chrono::system_clock::to_time_t(endB);
    std::cout << "Finished step B computation at " << std::ctime(&end_time)
              << "elapsed time: " << elapsed_seconds.count() << "s\n";

    // Step C
    // 1. Repeated A & B 100 times.
    // We can calculate this by checking the batch size directly.
    // Total time = (Time of step A + Time of step B) * 100 / BATCHSIZE_CPRA

    // 2. Pick the best, omitted here.
    // This step cost O(N), compared with other steps
    // this step's time consumption is usually negligible

    /*
        Step C omitted here
    */

    // Free 2D memory
    compute.deallocate(random_guess);
    compute.deallocate(t_random_guess_1);
    compute.deallocate(t_random_guess_2);
    compute.deallocate(dataConstr);
    compute.deallocate(spaceConstr);
    
     // 3D complex part test
    CPRA::Cpra<float, CPRA::IMPL_TYPE::CUDA> compute_3D(M, N, L, 1);
    std::complex<float>* random_guess_3D = (std::complex<float>*)compute_3D.allocate(sizeof(std::complex<float>) * M * N * L);
    std::complex<float>* complexDataConstr_3D = (std::complex<float>*)compute_3D.allocate(sizeof(std::complex<float>) * M * N * L);
    compute_3D.impl_->Initialize(reinterpret_cast<float*>(random_guess_3D), M * N * L * 2);
    compute_3D.impl_->Initialize(reinterpret_cast<float*>(complexDataConstr_3D), M * N * L * 2);

    float* spaceConstr_3D = (float*)compute_3D.allocate(sizeof(float) * M * N * L);
    compute_3D.impl_->Initialize(spaceConstr_3D, M * N * L);

    endC = std::chrono::system_clock::now();
    elapsed_seconds = (endB - start) * 100.0 / BATCHSIZE_CPRA;
    //end_time = std::chrono::system_clock::to_time_t(endC);
    std::cout << "Estimated step C elapsed time: " << elapsed_seconds.count() << "s\n";  

    // Step D, Reconstruct 3D object with complex phase
    // Num of iteration here is fixed at 50
    for(auto i = 0; i < 50; i++)
    {
        compute_3D.impl_->Forward3D(random_guess_3D);
        compute_3D.impl_->ComplexDataConstraint(random_guess_3D, complexDataConstr_3D, M * N * L, 1);
        compute_3D.impl_->Backward3D(random_guess_3D);
        compute_3D.impl_->SpaceConstraint(random_guess_3D, spaceConstr_3D, M * N * L, 1);
    }
    compute_3D.impl_->Sync();
    endD = std::chrono::system_clock::now();
    elapsed_seconds = endD - endC;
    end_time = std::chrono::system_clock::to_time_t(endD);
    std::cout << "Finished step D computation at " << std::ctime(&end_time)
              << "elapsed time: " << elapsed_seconds.count() << "s\n";

    // All time
    elapsed_seconds = endD - endC + (endB - start) * 100.0 / BATCHSIZE_CPRA;
    std::cout << "Estimated elapsed time with 100 initial guesses in total: " << elapsed_seconds.count() << "s\n";

    // Free 3D memory
    compute_3D.deallocate(random_guess_3D);
    compute_3D.deallocate(complexDataConstr_3D);
    compute_3D.deallocate(spaceConstr_3D);

    return;
}

void ShrinkWrap_Conventional_CUDA_Sample(int iter)
{
    CPRA::Cpra<float, CPRA::IMPL_TYPE::CUDA> compute(M, N, L, BATCHSIZE_CONV);
    std::complex<float>* random_guess = (std::complex<float>*)compute.allocate(sizeof(std::complex<float>) * M * N * L * BATCHSIZE_CONV);
    std::complex<float>* t_random_guess_1 = (std::complex<float>*)compute.allocate(sizeof(std::complex<float>) * M * N * L * BATCHSIZE_CONV);
    std::complex<float>* t_random_guess_2 = (std::complex<float>*)compute.allocate(sizeof(std::complex<float>) * M * N * L * BATCHSIZE_CONV);
    compute.impl_->Initialize(reinterpret_cast<float*>(random_guess), M * N * L * BATCHSIZE_CONV * 2);
    compute.impl_->Initialize(reinterpret_cast<float*>(t_random_guess_1), M * N * L * BATCHSIZE_CONV * 2);
    compute.impl_->Initialize(reinterpret_cast<float*>(t_random_guess_2), M * N * L * BATCHSIZE_CONV * 2);


    float* dataConstr = (float*)compute.allocate(sizeof(float) * M * N * L);
    compute.impl_->Initialize(dataConstr, M * N * L);
    float* spaceConstr = (float*)compute.allocate(sizeof(float) * M * N * L);
    compute.impl_->Initialize(spaceConstr, M * N * L);


    
    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();

    // Copy to temporary variable
    compute.impl_->Memcpy((void*)(t_random_guess_1), (void*)(random_guess), sizeof(std::complex<float>) * M * N * L * BATCHSIZE_CONV);
    compute.impl_->Memcpy((void*)(t_random_guess_2), (void*)(random_guess), sizeof(std::complex<float>) * M * N * L * BATCHSIZE_CONV);
    // Reconstruct
    for(auto i = 0; i < iter; i++) // each iteration
    {
        // Part 1
        compute.impl_->Forward2D(t_random_guess_1);
        compute.impl_->RealDataConstraint(t_random_guess_1, dataConstr, M * N * L * BATCHSIZE_CONV, BATCHSIZE_CONV);
        compute.impl_->Backward2D(t_random_guess_1);
        compute.impl_->MergeAddData(t_random_guess_1, random_guess, 1.0 + 1.0 / BETA, -1.0 / BETA, M * N * L * BATCHSIZE_CONV);
        compute.impl_->SpaceConstraint(t_random_guess_1, spaceConstr, M * N * L * BATCHSIZE_CONV, BATCHSIZE_CONV);
        
        // Part 2
        compute.impl_->SpaceConstraint(t_random_guess_2, spaceConstr, M * N * L * BATCHSIZE_CONV, BATCHSIZE_CONV);
        compute.impl_->MergeAddData(t_random_guess_2, random_guess, 1.0 - 1.0 / BETA, -1.0 / BETA, M * N * L * BATCHSIZE_CONV);
        compute.impl_->Forward2D(t_random_guess_2);
        compute.impl_->RealDataConstraint(t_random_guess_2, dataConstr, M * N * L * BATCHSIZE_CONV, BATCHSIZE_CONV);
        compute.impl_->Backward2D(t_random_guess_2);

        // Merge 
        compute.impl_->MergeAddData(t_random_guess_1, random_guess, BETA, 1.0, M * N * L * BATCHSIZE_CONV);
        compute.impl_->MergeAddData(t_random_guess_2, random_guess, -1.0 * BETA, 1.0, M * N * L * BATCHSIZE_CONV);
    }
    compute.impl_->Sync();
    //memory.impl_->Sync();
    end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    std::cout << "Finished computation at " << std::ctime(&end_time)
              << "elapsed time: " << elapsed_seconds.count() << "s\n"
              << "Estimated elapsed timewith 100 intial guesses in total : " 
              << elapsed_seconds.count() * 100 / BATCHSIZE_CONV << std::endl;

    // Free memory
    compute.deallocate(random_guess);
    compute.deallocate(t_random_guess_1);
    compute.deallocate(t_random_guess_2);
    compute.deallocate(dataConstr);
    compute.deallocate(spaceConstr);
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
        ShrinkWrap_CPRA_CUDA_Sample(epi, iter);
    }
    else if (argc == 2)
    {
        std::istringstream iter_ss(argv[1]);
        int iter;
        iter_ss >> iter;
        std::cout << "Running Conventional benchmark, iter is " << iter << std::endl;
        ShrinkWrap_Conventional_CUDA_Sample(iter);
    }
    else
    {
        std::cerr << "Wrong num of command line arg!" << std::endl;
    }
    return 0;
}