include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.12.1
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)

if(HAS_CUDA)
    add_executable(
        unitest_cuda
        unitest_cuda.cu
    )
    target_link_libraries(
        unitest_cuda PUBLIC
        GTest::gtest_main
    )
    target_link_libraries(
        unitest_cuda 
        PRIVATE 
        CUDA::cudart  
        CUDA::curand
        CUDA::cufft
    )
    gtest_discover_tests(unitest_cuda)
endif()

if(HAS_MKL)
    add_executable(
        unitest_mkl
        unitest_mkl.cpp
    )
    target_link_libraries(
        unitest_mkl PUBLIC
        GTest::gtest_main
    )
    
    if(HAS_OMP)
        target_link_libraries(unitest_mkl PRIVATE OpenMP::OpenMP_CXX_LIBRARIES)
    endif()

    target_include_directories(unitest_mkl PRIVATE "${MKLROOT}/include")
    target_link_directories(unitest_mkl PRIVATE "${MKLROOT}/lib/intel64")
    target_link_libraries(unitest_mkl
                            PRIVATE
                            mkl_intel_lp64
                            mkl_core
                            mkl_gnu_thread
                        )
    gtest_discover_tests(unitest_mkl)
endif()