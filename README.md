# Carousel-Phase-Retrieval-Algorithm
Carousel Phase Retrieval Algorithm (CPRA)

# System requirement
- Cmake minimum requirement 3.18
- To build CUDA and enable UVM oversubscription function, only LINUX is supported, otherwise WINDOWS is also supported.
- To build MKL version, please install Intel oneAPI toolkit, old Intel MKL toolkit can't be found by Cmake automatically (Can be added manually in CmakeLists.txt)

# Building instructions
- Don't build CUDA and MKL version together, because NVCC can't link MKL.

- To build and run CUDA unit test

```bash
mkdir build && cd build
cmake -DBUILD_CUDA=ON -DBUILD_MKL=OFF -DBUILD_TEST=ON
make
../bin/unitest_cuda
```

- To build and run CUDA benchmark with 10 episodes, 10 iterations within each episode

```bash
mkdir build && cd build
cmake -DBUILD_CUDA=ON -DBUILD_MKL=OFF -DBUILD_TEST=OFF
make
../bin/benchmark_cuda 10 10
```

- To build and run MKL unit test

```bash
mkdir build && cd build
cmake -DBUILD_CUDA=OFF -DBUILD_MKL=ON -DBUILD_TEST=ON
make
../bin/unitest_mkl
```

- To build and run MKL benchmark with 10 episodes, 10 iterations within each episode

```bash
mkdir build && cd build
cmake -DBUILD_CUDA=OFF -DBUILD_MKL=ON -DBUILD_TEST=OFF
make
../bin/benchmark_cuda 10 10
```
