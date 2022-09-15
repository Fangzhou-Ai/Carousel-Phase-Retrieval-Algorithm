#include "../include/CPRA.hpp"
#include <iostream>
#include <boost/program_options.hpp>
#include <sstream>
#include <chrono> 
#include <vector>
#include <string>
namespace po = boost::program_options;

std::complex<float>* ShrinkWrap_CPRA_CUDA_Sample(int M, int N, int L, int P, int BATCHSIZE_CPRA, 
                                                 float* dataConstr, float* spaceConstr, 
                                                 int epi, int iter, float Beta = 0.9)
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

    // Step A, pre-reconstruct
    // Num of iteration here is fixed at 1000
    // Shrinkwrap algo
    compute.impl_->Memcpy((void*)(t_random_guess_1), (void*)(random_guess), sizeof(std::complex<float>) * M * N * BATCHSIZE_CPRA);
    compute.impl_->Memcpy((void*)(t_random_guess_2), (void*)(random_guess), sizeof(std::complex<float>) * M * N * BATCHSIZE_CPRA);
    for(auto i = 0; i < 1000; i++)
    {
        // Part 1
        compute.impl_->Forward2D(t_random_guess_1);
        compute.impl_->DataConstraint(t_random_guess_1, dataConstr, M * N * BATCHSIZE_CPRA, BATCHSIZE_CPRA);
        compute.impl_->Backward2D(t_random_guess_1);
        // Sync here to make sure correct result
        //compute.impl_->Sync();
        compute.impl_->MergeAddData(t_random_guess_1, random_guess, 1.0 + 1.0 / Beta, -1.0 / Beta, M * N * BATCHSIZE_CPRA);
        compute.impl_->SpaceConstraint(t_random_guess_1, spaceConstr, M * N * BATCHSIZE_CPRA, BATCHSIZE_CPRA);
        
        // Part 2
        compute.impl_->SpaceConstraint(t_random_guess_2, spaceConstr, M * N * BATCHSIZE_CPRA, BATCHSIZE_CPRA);
        // Sync here to make sure correct result
        //compute.impl_->Sync();
        compute.impl_->MergeAddData(t_random_guess_2, random_guess, 1.0 - 1.0 / Beta, -1.0 / Beta, M * N * BATCHSIZE_CPRA);
        compute.impl_->Forward2D(t_random_guess_2);
        compute.impl_->DataConstraint(t_random_guess_2, dataConstr, M * N * BATCHSIZE_CPRA, BATCHSIZE_CPRA);
        compute.impl_->Backward2D(t_random_guess_2);

        // Merge 
        compute.impl_->MergeAddData(t_random_guess_1, random_guess, Beta, 1.0, M * N * BATCHSIZE_CPRA);
        //compute.impl_->Sync();
        compute.impl_->MergeAddData(t_random_guess_2, random_guess, -1.0 * Beta, 1.0, M * N * BATCHSIZE_CPRA);
        //compute.impl_->Sync();
    }
    compute.impl_->Sync();

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
                compute.impl_->DataConstraint(t_random_guess_1, dataConstr + M * N * p, M * N * BATCHSIZE_CPRA, BATCHSIZE_CPRA);
                compute.impl_->Backward2D(t_random_guess_1);
                // Sync here to make sure correct result
                //compute.impl_->Sync();
                compute.impl_->MergeAddData(t_random_guess_1, random_guess + M * N * BATCHSIZE_CPRA * p, 1.0 + 1.0 / Beta, -1.0 / Beta, M * N * BATCHSIZE_CPRA);
                compute.impl_->SpaceConstraint(t_random_guess_1, spaceConstr + M * N * p, M * N * BATCHSIZE_CPRA, BATCHSIZE_CPRA);
                
                // Part 2
                compute.impl_->SpaceConstraint(t_random_guess_2, spaceConstr + M * N * p, M * N * BATCHSIZE_CPRA, BATCHSIZE_CPRA);
                // Sync here to make sure correct result
                //compute.impl_->Sync();
                compute.impl_->MergeAddData(t_random_guess_2, random_guess + M * N * BATCHSIZE_CPRA * p, 1.0 - 1.0 / Beta, -1.0 / Beta, M * N * BATCHSIZE_CPRA);
                compute.impl_->Forward2D(t_random_guess_2);
                compute.impl_->DataConstraint(t_random_guess_2, dataConstr + M * N * p, M * N * BATCHSIZE_CPRA, BATCHSIZE_CPRA);
                compute.impl_->Backward2D(t_random_guess_2);

                // Merge 
                compute.impl_->MergeAddData(t_random_guess_1, random_guess + M * N * BATCHSIZE_CPRA * p, Beta, 1.0, M * N * BATCHSIZE_CPRA);
                //compute.impl_->Sync();
                compute.impl_->MergeAddData(t_random_guess_2, random_guess + M * N * BATCHSIZE_CPRA * p, -1.0 * Beta, 1.0, M * N * BATCHSIZE_CPRA);
                //compute.impl_->Sync();
            }
            
        }
    }
    compute.impl_->Sync();
    //memory.impl_->Sync();
    compute.deallocate(t_random_guess_1);
    compute.deallocate(t_random_guess_2);

    return random_guess;
}

int main(int argc, const char* argv[])
{
    // Params parser
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("M", po::value<int>(), "set first dimension")
        ("N", po::value<int>(), "set second dimension")
        ("L", po::value<int>(), "set third dimension")
        ("P", po::value<int>(), "set number of projected objects")
        ("Batch", po::value<int>(), "set batch size")
        ("Beta", po::value<float>(), "set shrinkwrap beta value")
        ("data_constraint", po::value<std::string>()->required(), "Data Constraint file")
        ("space_constraint", po::value<std::string>()->required(), "Space Constraint file")
        ("output_reconstruction", po::value<std::string>()->required(), "Output file for reconstructions")
        ("output_error", po::value<std::string>()->required(), "Output file for errors")
    ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm); 

    if (vm.count("help"))
    {
        std::cout << desc << std::endl;
        return 1;
    }
    // get parameters
    int M = vm["M"].as<int>();
    int N = vm["N"].as<int>();
    int L = vm["L"].as<int>();
    int P = vm["P"].as<int>();
    int Batch = vm["Batch"].as<int>();
    float Beta = vm["Beta"].as<float>();
    const std::string data_constraint = vm["data_constraint"].as<std::string>();
    const std::string space_constraint = vm["space_constraint"].as<std::string>();
    const std::string output_reconstruction = vm["output_reconstruction"].as<std::string>();
    const std::string output_error = vm["output_error"].as<std::string>();
    // Output parameters
    std::cout << "CPRA CUDA implementiation, Fangzhou Ai @ Prof. Vitaliy Lomakin's group UCSD." << std::endl;
    std::cout << "Object dimension is " << M << ", " << N << ", " << L << "." << std::endl;
    std::cout << "Number of projected object is " << P << "." << std::endl;
    std::cout << "Batch size is " << Batch << "." << std::endl;
    std::cout << "Shrinkwrap hyper-param beta is " << Beta << "." << std::endl;
    std::cout << "Data constraint file is " << data_constraint << "." << std::endl;
    std::cout << "Space constraint file is " << space_constraint << "." << std::endl;
    std::cout << "Output reconstructions file is " << output_reconstruction << "." << std::endl;
    std::cout << "Output errors file is " << output_error << "." << std::endl;
    return EXIT_SUCCESS;
}