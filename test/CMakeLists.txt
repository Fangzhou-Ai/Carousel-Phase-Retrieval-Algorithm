include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.12.1
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(
    unitest
    unitest.cpp
)
target_link_libraries(
    unitest PUBLIC
    GTest::gtest_main
)

if(HAS_CUDA)
    target_link_libraries(
        unitest 
        PRIVATE 
        CUDA::cudart  
        CUDA::curand
    )
endif()

if(HAS_MKL)
target_include_directories(unitest PRIVATE "${MKLROOT}/include")
target_link_directories(unitest PRIVATE "${MKLROOT}/lib/intel64")
target_link_libraries(unitest
                      PRIVATE
                      mkl_intel_lp64
                      mkl_core
                      mkl_gnu_thread
                     )
endif()


if(HAS_OMP)
target_link_libraries(unitest PRIVATE OpenMP::OpenMP_CXX)
endif()

include(GoogleTest)

gtest_discover_tests(unitest)