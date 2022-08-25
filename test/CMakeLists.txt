include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.12.1
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(
    temp_test
    temp_test.cpp
)
target_link_libraries(
    temp_test PUBLIC
    GTest::gtest_main
)

if(HAS_CUDA)
    target_link_libraries(
        temp_test 
        PRIVATE 
        CUDA::cudart  
        CUDA::curand
    )
endif()

if(HAS_MKL)
target_include_directories(temp_test PRIVATE "${MKLROOT}/include")    
target_link_directories(temp_test PRIVATE "${MKLROOT}/lib/intel64")
endif()


include(GoogleTest)

gtest_discover_tests(temp_test)