#pragma once
#include <iostream>
#include <boost/program_options.hpp>
#include <sstream>
#include <chrono> 
#include <vector>
#include <string>
#include <stdlib.h>

namespace po = boost::program_options;


namespace CPRA{

class Parser
{
    public:
        explicit Parser(std::string desc = "Allowed options:") : desc_(desc)
        {
            // Define parser
            desc_.add_options()
                ("help,h", "produce help message")
                ("M,m", po::value<int>(&M)->default_value(160), "set first dimension")
                ("N,n", po::value<int>(&N)->default_value(160), "set second dimension")
                ("L,l", po::value<int>(&L)->default_value(160), "set third dimension")
                ("P,p", po::value<int>(&P)->default_value(90), "set number of projected objects")
                ("epi,e", po::value<int>(&Epi)->default_value(1), "set number of episodes")
                ("iter,i", po::value<int>(&Iter)->default_value(1), "set number of iterations within each episodes")
                ("preit", po::value<int>(&PreIter)->default_value(1000), "set number of iterations for pre-reconstruction")
                ("Batch", po::value<int>(&Batch)->default_value(1), "set batch size")
                ("Total", po::value<int>(&Total)->default_value(100), "Total batch size")
                ("Beta", po::value<float>(&Beta)->default_value(0.9), "set shrinkwrap beta value")
                ("data_constraint,d", po::value<std::string>(&data_constraint)->default_value("./data_constraint.bin"), "Data Constraint file")
                ("space_constraint,s", po::value<std::string>(&space_constraint)->default_value("./space_constraint.bin"), "Space Constraint file")
                ("output_reconstruction,r", po::value<std::string>(&output_reconstruction)->default_value("./output_reconstruction.bin"), "Output file for reconstructions")
                ("output_error", po::value<std::string>(&output_error)->default_value("./output_error.bin"), "Output file for errors")
            ;
        }

        void parse(int argc, const char* argv[])
        {
            po::variables_map vm;
            po::store(po::parse_command_line(argc, argv, desc_), vm);
            po::notify(vm); 

            if (vm.count("help"))
            {
                std::cout << desc_ << std::endl;
                exit(EXIT_SUCCESS);
            }

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
            
            return;
        }

        int getM(){return M;}
        int getN(){return N;}
        int getL(){return L;}
        int getP(){return P;}
        int getEpi(){return Epi;}
        int getIter(){return Iter;}
        int getPreIter(){return PreIter;}
        int getBatch(){return Batch;}
        int getTotal(){return Total;}
        float getBeta(){return Beta;}
        std::string getDataConstr(){return data_constraint;}
        std::string getSpaceConstr(){return space_constraint;}
        std::string getOutputReconstr(){return output_reconstruction;}
        std::string getOutputError(){return output_error;}
        ~Parser(){}
    private:
        po::options_description desc_;
        // Parameters
        int M, N, L, P, Epi, Iter, PreIter, Batch, Total;
        float Beta;
        std::string data_constraint, space_constraint, output_reconstruction, output_error;
};


}
