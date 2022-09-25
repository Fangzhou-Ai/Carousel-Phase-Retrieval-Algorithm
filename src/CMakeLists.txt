find_package(Boost COMPONENTS program_options REQUIRED)
include_directories(${Boost_INCLUDE_DIR})

if(HAS_CUDA)
    add_executable(
            benchmark_cuda
            benchmark_cuda.cu
        )
    target_link_libraries(
        benchmark_cuda 
        PRIVATE 
        CUDA::cudart  
        CUDA::curand
        CUDA::cufft
        CUDA::cublas
    )
    target_compile_options(benchmark_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --generate-line-info
                       --use_fast_math
                       >)

    add_executable(
            CPRA_CUDA
            CPRA_CUDA.cu
        )
    target_link_libraries(
        CPRA_CUDA 
        PRIVATE 
        CUDA::cudart  
        CUDA::curand
        CUDA::cufft
        CUDA::cublas
        ${Boost_LIBRARIES}
    )
    target_compile_options(CPRA_CUDA PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --generate-line-info
                       --use_fast_math
                       >)

    add_executable(
    CONV_CUDA
    CONV_CUDA.cu
        )
    target_link_libraries(
        CONV_CUDA 
        PRIVATE 
        CUDA::cudart  
        CUDA::curand
        CUDA::cufft
        CUDA::cublas
        ${Boost_LIBRARIES}
    )
    target_compile_options(CONV_CUDA PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                        --generate-line-info
                        --use_fast_math
                        >)

endif()

if(HAS_MKL)
    add_executable(
        benchmark_mkl
        benchmark_mkl.cpp
    )
    if(HAS_OMP)
        target_link_libraries(benchmark_mkl PRIVATE OpenMP::OpenMP_CXX_LIBRARIES)
    endif()

    target_include_directories(benchmark_mkl PRIVATE "${MKLROOT}/include")
    target_link_directories(benchmark_mkl PRIVATE "${MKLROOT}/lib/intel64")
    target_link_libraries(benchmark_mkl
                            PRIVATE
                            mkl_intel_lp64
                            mkl_core
                            mkl_gnu_thread
                        )
endif()