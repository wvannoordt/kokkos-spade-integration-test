find_package(Kokkos REQUIRED)
# set(Kokkos_ROOT /home/amovsesy/spack/opt/spack/linux-rhel8-zen3/gcc-11.3.0/kokkos-3.7.01-75nmoutfjze35jd7frgtzbswxq63y2v6/lib64/cmake/Kokkos)
# set(CMAKE_CXX_COMPILER /home/amovsesy/spack/opt/spack/linux-rhel8-zen3/gcc-11.3.0/kokkos-nvcc-wrapper-3.2.00-i5v6unjdblj2y45pfo7tskjjk5f6m66u/bin/nvcc_wrapper)
set(Kokkos_ENABLE_CUDA OFF)
set(Kokkos_ENABLE_OPENMP ON)
set(KOKKOS_DEVICES Serial)
add_executable(test main.cc)

set(CMAKE_CXX_COMPILER mpicxx)

set(SCIDF /home/amovsesy/scratch/spade/scidf)
set(SPADE /home/amovsesy/scratch/spade/spade)

set(CMAKE_CXX_FLAGS "-fconcepts-diagnostics-depth=3 -fmax-errors=4 -Wterminate -std=c++20 -O3 -I${SCIDF}/src -I${SPADE}/src")

include_directories(${SCIDF}/src)
include_directories(${SPADE}/src)

target_link_libraries(test Kokkos::kokkos)

