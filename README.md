# Carousel-Phase-Retrieval-Algorithm
Carousel Phase Retrieval Algorithm (CPRA)

# System requirement
- Cmake minimum requirement 3.18
- To build CUDA and enable UVM oversubscription function, only LINUX is supported.
- To build MKL version, please install Intel oneAPI toolkit, old Intel MKL toolkit can't be found by Cmake automatically (Can be added manually in CmakeLists.txt)
- Boost

# Docker
- We offer a docker image that meet all system requirements we mentioned above, to download the image, please `docker pull ucsdcem/carousel-phase-retrieval-algorithm:11.8.0_2024.1-HPC_Boost-1.84.0_Devel_Ubuntu-20.04`

# Building instructions
- Don't build CUDA and MKL version together, because NVCC can't link MKL.

- To build and run CUDA unit test

```bash
cd Carousel-Phase-Retrieval-Algorithm
mkdir build && cd build
cmake -DHAS_CUDA=ON -DHAS_MKL=OFF -DBUILD_TEST=ON ../
make
../bin/unitest_cuda
```

- To build and run CUDA benchmark with 10 episodes, 10 iterations within each episode

```bash
cd Carousel-Phase-Retrieval-Algorithm
mkdir build && cd build
cmake -DHAS_CUDA=ON -DHAS_MKL=OFF -DBUILD_TEST=OFF ../
make
../bin/benchmark_cuda 10 10
```

- To build and run MKL unit test

```bash
cd Carousel-Phase-Retrieval-Algorithm
mkdir build && cd build
cmake -DHAS_CUDA=OFF -DHAS_MKL=ON -DBUILD_TEST=ON ../
make
../bin/unitest_mkl
```

- To build and run MKL benchmark with 10 episodes, 10 iterations within each episode

```bash
cd Carousel-Phase-Retrieval-Algorithm
mkdir build && cd build
cmake -DHAS_CUDA=OFF -DHAS_MKL=ON -DBUILD_TEST=OFF ../
make
../bin/benchmark_cuda 10 10
```
