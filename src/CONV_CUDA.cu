#include "../include/CPRA.hpp"
#include "../include/Command_Parser.hpp"
#include <iostream>
#include <sstream>
#include <chrono> 
#include <vector>
#include <string>
#include <utility>

template<typename T>
std::pair<std::complex<T>*, T*> ShrinkWrap_CONV_CUDA_Sample(int M, int N, int L, int BATCHSIZE_CONV, 
                                                            T* dataConstr, T* spaceConstr, 
                                                            int iter, T Beta = 0.9)
{
    // 2D part test
    // Shrinkwrap
    CPRA::Cpra<T, CPRA::IMPL_TYPE::CUDA> compute(M, N, L, BATCHSIZE_CONV);
    std::complex<T>* random_guess = (std::complex<T>*)compute.allocate(sizeof(std::complex<T>) * M * N * L * BATCHSIZE_CONV);
    std::complex<T>* t_random_guess_1 = (std::complex<T>*)compute.allocate(sizeof(std::complex<T>) * M * N * L * BATCHSIZE_CONV);
    std::complex<T>* t_random_guess_2 = (std::complex<T>*)compute.allocate(sizeof(std::complex<T>) * M * N * L * BATCHSIZE_CONV);
    compute.impl_->Initialize(reinterpret_cast<T*>(random_guess), M * N * L * BATCHSIZE_CONV * 2);
    compute.impl_->Initialize(reinterpret_cast<T*>(t_random_guess_1), M * N * L * BATCHSIZE_CONV * 2);
    compute.impl_->Initialize(reinterpret_cast<T*>(t_random_guess_2), M * N * L * BATCHSIZE_CONV * 2);

    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();

    // For error control
    std::complex<T>* old_record = (std::complex<T>*)compute.allocate(sizeof(std::complex<T>) * M * N * L * BATCHSIZE_CONV);
    T* error = (T*)compute.allocate(sizeof(T) * BATCHSIZE_CONV);

    // Copy to temporary variable
    compute.impl_->Memcpy((void*)(t_random_guess_1), (void*)random_guess, sizeof(std::complex<T>) * M * N * L * BATCHSIZE_CONV);
    compute.impl_->Memcpy((void*)(t_random_guess_2), (void*)random_guess, sizeof(std::complex<T>) * M * N * L * BATCHSIZE_CONV);
    // Reconstruct
    // Shrinkwrap algo
    for(auto i = 0; i < iter; i++) // each iteration
    {
        // error part
        if(i == iter - 1)
        {
            compute.impl_->Memcpy((void*)old_record,
                                  (void*)random_guess,
                                  sizeof(std::complex<T>) * M * N * L * BATCHSIZE_CONV);
        }
            
        // Part 1
        compute.impl_->Forward3D(t_random_guess_1);
        compute.impl_->DataConstraint(t_random_guess_1, dataConstr, M * N * L * BATCHSIZE_CONV, BATCHSIZE_CONV);
        compute.impl_->Backward3D(t_random_guess_1);
        compute.impl_->MergeAddData(random_guess, t_random_guess_1, -1.0 / Beta, 1.0 + 1.0 / Beta, M * N * L * BATCHSIZE_CONV);
        compute.impl_->SpaceConstraint(t_random_guess_1, spaceConstr, M * N * L * BATCHSIZE_CONV, BATCHSIZE_CONV);
        
        // Part 2
        compute.impl_->SpaceConstraint(t_random_guess_2, spaceConstr, M * N * L * BATCHSIZE_CONV, BATCHSIZE_CONV);
        compute.impl_->MergeAddData(random_guess, t_random_guess_2, 1.0 / Beta, 1.0 - 1.0 / Beta, M * N * L * BATCHSIZE_CONV);
        compute.impl_->Forward3D(t_random_guess_2);
        compute.impl_->DataConstraint(t_random_guess_2, dataConstr, M * N * L * BATCHSIZE_CONV, BATCHSIZE_CONV);
        compute.impl_->Backward3D(t_random_guess_2);

        // Merge 
        compute.impl_->MergeAddData(t_random_guess_1, random_guess, Beta, 1.0, M * N * L * BATCHSIZE_CONV);
        compute.impl_->MergeAddData(t_random_guess_2, random_guess, -1.0 * Beta, 1.0, M * N * L * BATCHSIZE_CONV);

        // error part
        if(i == iter - 1)
        {
            compute.impl_->Memcpy((void*)t_random_guess_1,
                                  (void*)random_guess,
                                  sizeof(std::complex<T>) * M * N * L * BATCHSIZE_CONV);
            compute.impl_->Forward3D(old_record);
            compute.impl_->Forward3D(t_random_guess_1);
            compute.impl_->ConvergeError(old_record, t_random_guess_1, error, M, N, L, BATCHSIZE_CONV);
        }
    }

    // Finishing reconstruction
    compute.impl_->Memcpy((void*)t_random_guess_1,
                          (void*)random_guess,
                          sizeof(std::complex<T>) * M * N * L * BATCHSIZE_CONV);
    compute.impl_->Forward3D(t_random_guess_1);
    compute.impl_->DataConstraint(t_random_guess_1, dataConstr, M * N * L * BATCHSIZE_CONV, BATCHSIZE_CONV);
    compute.impl_->Backward3D(t_random_guess_1);
    compute.impl_->MergeAddData(random_guess, t_random_guess_1, -1.0 / Beta, 1.0 + 1.0 / Beta, M * N * L * BATCHSIZE_CONV);
    compute.impl_->SpaceConstraint(t_random_guess_1, spaceConstr, M * N * L * BATCHSIZE_CONV, BATCHSIZE_CONV);
    compute.impl_->Memcpy((void*)random_guess,
                          (void*)t_random_guess_1,
                          sizeof(std::complex<T>) * M * N * L * BATCHSIZE_CONV);

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

int main(int argc, const char* argv[])
{
    CPRA::Parser parser;
    parser.parse(argc, argv);
    // an obj for data preparation
    CPRA::Cpra<float, CPRA::IMPL_TYPE::CUDA> obj(1, 1, 1, 1);
    float* dataConstr = (float*)obj.allocate(sizeof(float) * parser.getM() * parser.getN() * parser.getL());
    if(!obj.ReadMatrixFromFile(parser.getDataConstr(), dataConstr, parser.getM(), parser.getN(), parser.getL()))
    {
        throw std::runtime_error("Read file " + parser.getDataConstr() + " failed!");
    }
    
    float* spaceConstr = (float*)obj.allocate(sizeof(float) * parser.getM() * parser.getN() * parser.getL());
    if(!obj.ReadMatrixFromFile(parser.getSpaceConstr(), spaceConstr, parser.getM(), parser.getN(), parser.getL()))
    {
        throw std::runtime_error("Read file " + parser.getSpaceConstr() + " failed!");
    }
    uint64_t N = std::max(parser.getBatch(), parser.getTotal());
    int cnt = N;
    int fileCnt = 1; // count file name index
    float* per_real_rec_projected_object = (float*)obj.allocate(sizeof(float) * parser.getM() * parser.getN() * parser.getL());
    float* per_error = (float*)obj.allocate(sizeof(float));
    while(true)
    {
        int n = std::min(cnt, parser.getBatch());
        std::cout << "Reconstrucint batch size " << n << ". Total batch left " << cnt << "." << std::endl;
        auto results = ShrinkWrap_CONV_CUDA_Sample<float>(parser.getM(),
                                                          parser.getN(),
                                                          parser.getL(),
                                                          n,
                                                          dataConstr,
                                                          spaceConstr,
                                                          parser.getIter(),
                                                          parser.getBeta()
                                                         );
        auto rec_projected_object = results.first;
        auto error = results.second;
        uint64_t num = parser.getM() * parser.getN() * parser.getL() * n;
        float* real_rec_projected_object = (float*)obj.allocate(sizeof(float) * num);
        obj.ComplexToReal(rec_projected_object, real_rec_projected_object, num);
        uint64_t start = N - cnt;
        // Write data
        for(uint64_t i = 0; i < n; i++)
        {
            per_error[i] = error[i];
            obj.impl_->Memcpy(per_real_rec_projected_object,
                              real_rec_projected_object + i * parser.getM() * parser.getN() * parser.getL(),
                              sizeof(float) * parser.getM() * parser.getN() * parser.getL());
                    
        }
        obj.WriteMatrixToFile(parser.getOutputReconstr()+"."+std::to_string(fileCnt),
                                per_real_rec_projected_object,
                                parser.getM() * parser.getN() * parser.getL(), 1, 1);
        obj.WriteMatrixToFile(parser.getOutputError()+"."+std::to_string(fileCnt),
                                per_error,
                                sizeof(float), 1, 1);
        fileCnt++;

        obj.deallocate(rec_projected_object);
        obj.deallocate(real_rec_projected_object);
        obj.deallocate(error);

        cnt -= n;
        if(cnt <= 0) break;  
    }
    // free memory
    obj.deallocate(per_real_rec_projected_object);
    obj.deallocate(per_error);
    obj.deallocate(dataConstr);
    obj.deallocate(spaceConstr);
    return EXIT_SUCCESS;
}