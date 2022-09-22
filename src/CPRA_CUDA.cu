#include "../include/CPRA.hpp"
#include "../include/Command_Parser.hpp"
#include <iostream>
#include <sstream>
#include <chrono> 
#include <vector>
#include <string>
#include <utility>

template<typename T>
std::pair<std::complex<T>*, T*> ShrinkWrap_CPRA_CUDA_Sample(int M, int N, int L, int P, int BATCHSIZE_CPRA, 
                                                                T* dataConstr, T* spaceConstr, 
                                                                int epi, int iter, T Beta = 0.9)
{
    // 2D part test
    // Shrinkwrap
    CPRA::Cpra<T, CPRA::IMPL_TYPE::CUDA> compute(M, N, L, BATCHSIZE_CPRA);
    std::complex<T>* random_guess = (std::complex<T>*)compute.allocate(sizeof(std::complex<T>) * M * N * P * BATCHSIZE_CPRA);
    std::complex<T>* t_random_guess_1 = (std::complex<T>*)compute.allocate(sizeof(std::complex<T>) * M * N * BATCHSIZE_CPRA);
    std::complex<T>* t_random_guess_2 = (std::complex<T>*)compute.allocate(sizeof(std::complex<T>) * M * N * BATCHSIZE_CPRA);
    compute.impl_->Initialize(reinterpret_cast<T*>(random_guess), M * N * P * BATCHSIZE_CPRA * 2);
    compute.impl_->Initialize(reinterpret_cast<T*>(t_random_guess_1), M * N * BATCHSIZE_CPRA * 2);
    compute.impl_->Initialize(reinterpret_cast<T*>(t_random_guess_2), M * N * BATCHSIZE_CPRA * 2);

    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();

    // For error control
    std::complex<T>* old_record = (std::complex<T>*)compute.allocate(sizeof(std::complex<T>) * M * N * BATCHSIZE_CPRA);
    T* error = (T*)compute.allocate(sizeof(T) * P * BATCHSIZE_CPRA);

    // Step A, pre-reconstruct
    // Num of iteration here is fixed at 1000
    // Shrinkwrap algo
    compute.impl_->Memcpy((void*)(t_random_guess_1), (void*)(random_guess), sizeof(std::complex<T>) * M * N * BATCHSIZE_CPRA);
    compute.impl_->Memcpy((void*)(t_random_guess_2), (void*)(random_guess), sizeof(std::complex<T>) * M * N * BATCHSIZE_CPRA);
    for(auto i = 0; i < 1000; i++)
    {
        // Part 1
        compute.impl_->Forward2D(t_random_guess_1);
        compute.impl_->DataConstraint(t_random_guess_1, dataConstr, M * N * BATCHSIZE_CPRA, BATCHSIZE_CPRA);
        compute.impl_->Backward2D(t_random_guess_1);
        compute.impl_->MergeAddData(random_guess, t_random_guess_1, -1.0 / Beta, 1.0 + 1.0 / Beta, M * N * BATCHSIZE_CPRA);
        compute.impl_->SpaceConstraint(t_random_guess_1, spaceConstr, M * N * BATCHSIZE_CPRA, BATCHSIZE_CPRA);
        
        // Part 2
        compute.impl_->SpaceConstraint(t_random_guess_2, spaceConstr, M * N * BATCHSIZE_CPRA, BATCHSIZE_CPRA);
        compute.impl_->MergeAddData(random_guess, t_random_guess_2, 1.0 / Beta, 1.0 - 1.0 / Beta, M * N * BATCHSIZE_CPRA);
        compute.impl_->Forward2D(t_random_guess_2);
        compute.impl_->DataConstraint(t_random_guess_2, dataConstr, M * N * BATCHSIZE_CPRA, BATCHSIZE_CPRA);
        compute.impl_->Backward2D(t_random_guess_2);
        // Merge 
        compute.impl_->MergeAddData(t_random_guess_1, random_guess, Beta, 1.0, M * N * BATCHSIZE_CPRA);
        compute.impl_->MergeAddData(t_random_guess_2, random_guess, -1.0 * Beta, 1.0, M * N * BATCHSIZE_CPRA);
    }
    compute.impl_->Sync();

    // Step B, reconstruct 2D projected object
    for(auto e = 0; e < epi; e++)  // episode
    {
        for(auto p = 0; p < P; p++) // each projected object
        {
           // Merge data
           if(p == P - 1)
           {
                compute.impl_->MergeAddData(random_guess + M * N * BATCHSIZE_CPRA * (p - 1), random_guess + M * N * BATCHSIZE_CPRA * p, 0.5, 0.5, M * N * BATCHSIZE_CPRA);  
           }
           else if(p == 0)
           {
                compute.impl_->MergeAddData(random_guess + M * N * BATCHSIZE_CPRA * (p + 1), random_guess + M * N * BATCHSIZE_CPRA * p, 0.5, 0.5, M * N * BATCHSIZE_CPRA);
           }
           else
           {
                compute.impl_->MergeAddData(random_guess + M * N * BATCHSIZE_CPRA * (p - 1), random_guess + M * N * BATCHSIZE_CPRA * p, 0.5, 0, M * N * BATCHSIZE_CPRA); 
                compute.impl_->MergeAddData(random_guess + M * N * BATCHSIZE_CPRA * (p + 1), random_guess + M * N * BATCHSIZE_CPRA * p, 0.5, 1, M * N * BATCHSIZE_CPRA);
           }

            // Copy to temporary variable
            compute.impl_->Memcpy((void*)(t_random_guess_1), (void*)(random_guess + M * N * BATCHSIZE_CPRA * p), sizeof(std::complex<T>) * M * N * BATCHSIZE_CPRA);
            compute.impl_->Memcpy((void*)(t_random_guess_2), (void*)(random_guess + M * N * BATCHSIZE_CPRA * p), sizeof(std::complex<T>) * M * N * BATCHSIZE_CPRA);
            // Reconstruct
            // Shrinkwrap algo
            for(auto i = 0; i < iter; i++) // each iteration
            {
                // error part
                if(i == iter - 1 && e == epi - 1)
                {
                    compute.impl_->Memcpy((void*)old_record,
                                          (void*)(random_guess + M * N * BATCHSIZE_CPRA * p),
                                          sizeof(std::complex<T>) * M * N * BATCHSIZE_CPRA);
                }
                    
                
                // Part 1
                compute.impl_->Forward2D(t_random_guess_1);
                compute.impl_->DataConstraint(t_random_guess_1, dataConstr + M * N * p, M * N * BATCHSIZE_CPRA, BATCHSIZE_CPRA);
                compute.impl_->Backward2D(t_random_guess_1);
                compute.impl_->MergeAddData(random_guess + M * N * BATCHSIZE_CPRA * p, t_random_guess_1, -1.0 / Beta, 1.0 + 1.0 / Beta, M * N * BATCHSIZE_CPRA);
                compute.impl_->SpaceConstraint(t_random_guess_1, spaceConstr + M * N * p, M * N * BATCHSIZE_CPRA, BATCHSIZE_CPRA);
                
                // Part 2
                compute.impl_->SpaceConstraint(t_random_guess_2, spaceConstr + M * N * p, M * N * BATCHSIZE_CPRA, BATCHSIZE_CPRA);
                compute.impl_->MergeAddData(random_guess + M * N * BATCHSIZE_CPRA * p, t_random_guess_2, 1.0 / Beta, 1.0 - 1.0 / Beta, M * N * BATCHSIZE_CPRA);
                compute.impl_->Forward2D(t_random_guess_2);
                compute.impl_->DataConstraint(t_random_guess_2, dataConstr + M * N * p, M * N * BATCHSIZE_CPRA, BATCHSIZE_CPRA);
                compute.impl_->Backward2D(t_random_guess_2);

                // Merge 
                compute.impl_->MergeAddData(t_random_guess_1, random_guess + M * N * BATCHSIZE_CPRA * p, Beta, 1.0, M * N * BATCHSIZE_CPRA);
                compute.impl_->MergeAddData(t_random_guess_2, random_guess + M * N * BATCHSIZE_CPRA * p, -1.0 * Beta, 1.0, M * N * BATCHSIZE_CPRA);

                // error part
                if(i == iter - 1 && e == epi - 1)
                {
                    compute.impl_->Memcpy((void*)t_random_guess_1,
                                          (void*)(random_guess + M * N * BATCHSIZE_CPRA * p),
                                          sizeof(std::complex<T>) * M * N * BATCHSIZE_CPRA);
                    compute.impl_->Forward2D(old_record);
                    compute.impl_->Forward2D(t_random_guess_1);
                    compute.impl_->ConvergeError(old_record, t_random_guess_1, error + BATCHSIZE_CPRA * p, M, N, 1, BATCHSIZE_CPRA);
                }
            }

            for(auto p = 0; p < P; p++)
            {
                // Finishing reconstruction
                compute.impl_->Memcpy((void*)t_random_guess_1,
                                      (void*)(random_guess + M * N * BATCHSIZE_CPRA * p),
                                      sizeof(std::complex<T>) * M * N * BATCHSIZE_CPRA);
                compute.impl_->Forward2D(t_random_guess_1);
                compute.impl_->DataConstraint(t_random_guess_1, dataConstr + M * N * p, M * N * BATCHSIZE_CPRA, BATCHSIZE_CPRA);
                compute.impl_->Backward2D(t_random_guess_1);
                compute.impl_->MergeAddData(random_guess + M * N * BATCHSIZE_CPRA * p, t_random_guess_1, -1.0 / Beta, 1.0 + 1.0 / Beta, M * N * BATCHSIZE_CPRA);
                compute.impl_->SpaceConstraint(t_random_guess_1, spaceConstr + M * N * p, M * N * BATCHSIZE_CPRA, BATCHSIZE_CPRA);
                compute.impl_->Memcpy((void*)(random_guess + M * N * BATCHSIZE_CPRA * p),
                                      (void*)t_random_guess_1,
                                      sizeof(std::complex<T>) * M * N * BATCHSIZE_CPRA);
            }
            
        }
    }
    compute.impl_->Sync();
    end = std::chrono::system_clock::now();
    std::chrono::duration<float> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    std::cout << "Finished computation at " << std::ctime(&end_time)
              << "elapsed time: " << elapsed_seconds.count() << "s\n";

    compute.deallocate(t_random_guess_1);
    compute.deallocate(t_random_guess_2);
    compute.deallocate(old_record);
    return std::make_pair(random_guess, error);
}

// Tune it by yourself if needed
#define TOTAL_BATCH 100

int main(int argc, const char* argv[])
{
    CPRA::Parser parser;
    parser.parse(argc, argv);
    // an obj for data preparation
    CPRA::Cpra<float, CPRA::IMPL_TYPE::CUDA> obj(1, 1, 1, 1);
    float* dataConstr = (float*)obj.allocate(sizeof(float) * parser.getM() * parser.getN() * parser.getP());
    if(!obj.ReadMatrixFromFile(parser.getDataConstr(), dataConstr, parser.getM(), parser.getN(), parser.getP()))
    {
        throw std::runtime_error("Read file " + parser.getDataConstr() + " failed!");
    }
    
    float* spaceConstr = (float*)obj.allocate(sizeof(float) * parser.getM() * parser.getN() * parser.getP());
    if(!obj.ReadMatrixFromFile(parser.getSpaceConstr(), spaceConstr, parser.getM(), parser.getN(), parser.getP()))
    {
        throw std::runtime_error("Read file " + parser.getSpaceConstr() + " failed!");
    }
    int N = std::max(parser.getBatch(), TOTAL_BATCH);
    int cnt = N;
    float* real_rec_projected_objects = (float*)obj.allocate(sizeof(float) * parser.getM() * parser.getN() * parser.getP() * N);
    float* errors = (float*)obj.allocate(sizeof(float) * parser.getP() * N);
    while(true)
    {
        int n = std::min(cnt, parser.getBatch());
        std::cout << "Reconstrucint batch size " << n << ". Total batch left " << cnt << "." << std::endl;
        auto results = ShrinkWrap_CPRA_CUDA_Sample<float>(parser.getM(),
                                                        parser.getN(),
                                                        parser.getL(),
                                                        parser.getP(),
                                                        n,
                                                        dataConstr,
                                                        spaceConstr,
                                                        parser.getEpi(),
                                                        parser.getIter(),
                                                        parser.getBeta()
                                                        );
        auto rec_projected_object = results.first;
        auto error = results.second;
        uint64_t num = parser.getM() * parser.getN() * parser.getP() * n;
        float* real_rec_projected_object = (float*)obj.allocate(sizeof(float) * num);
        obj.ComplexToReal(rec_projected_object, real_rec_projected_object, num);
        int start = N - cnt;
        // Copy data to buffer
        for(auto i = 0; i < parser.getP(); i++)
        {
            // copy object
            obj.impl_->Memcpy(real_rec_projected_objects + (i * TOTAL_BATCH + start) * parser.getM() * parser.getN(),
                              real_rec_projected_object + (i * n) * parser.getM() * parser.getN(),
                              sizeof(float) * parser.getM() * parser.getN() * n);
            obj.impl_->Memcpy(errors + i * TOTAL_BATCH + start,
                              error + i * n,
                              sizeof(float) * n);
        }

        obj.deallocate(rec_projected_object);
        obj.deallocate(real_rec_projected_object);
        obj.deallocate(error);

        cnt -= n;
        if(cnt <= 0) break;  
    }
    //write file
    obj.WriteMatrixToFile(parser.getOutputReconstr(), real_rec_projected_objects, parser.getM() * parser.getN() * parser.getP() * N, 1, 1);
    obj.WriteMatrixToFile(parser.getOutputError(), errors, parser.getP() * N, 1, 1);
    // free memory
    obj.deallocate(dataConstr);
    obj.deallocate(spaceConstr);
    obj.deallocate(real_rec_projected_objects);
    obj.deallocate(errors);
    return EXIT_SUCCESS;
}