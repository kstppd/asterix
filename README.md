This repository contains the compact Multi-Layer-Perceptron (TinyAI) and bindings developed to enable efficient lossy compression in [Vlasiator](https://github.com/fmihpc/vlasiator). 

+ TinyAI can be deployed on either CPUs or GPUs.
+ TinyAI uses a custom matrix implementation which is hosted in this repository as well. 
+ TinyAI uses a thread safe memory pool ( which resembles a free-list allocator) to perform all internal allocations. The memory pool is also hosted in this repository under ```include```.

## Installation
If you plan to use TinyAI at you will need a system with a dedicated GPU card, either NVIDIA or AMD and an installation of CUDA or ROCm. TinyAI rerquires at least ```cuda-11.0``` or ```rocm 5.4```. To run the tests ```googletest``` ```spdlog``` and ```stbimage``` need to be installed on your system.

## Run the tests (supported only for NVIDIA)
```
cd asterix/  
mkdir subprojects
meson wrap install gtest
meson wrap install libcurl
meson wrap install spdlog 
meson setup build  --buildtype=release
meson compile -C build --jobs=8
meson test -C build
```

## Examples
For examples of how TinyAI's MLP can be used the tests under ```tests``` are descriptive enough to provide guidance.

## Acknowledgments
This software library was built within project Adaptive Strategies Towards Expedient Recovery In eXascale (ASTERIX) at University of Helsinki and at CSC – IT Center for Science Ltd. The Innovation Study ASTERIX has received funding through the Inno4scale project, which is funded by the European High-Performance Computing Joint Undertaking (JU) under Grant Agreement No 101118139. The JU receives support from the European Union's Horizon Europe Programme. 
