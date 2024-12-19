#!/bin/bash

set -e   # Abort on error

WORKSPACE=`pwd`

if [[ z$1 != "z" ]]; then
   PLATFORM=-$1
else
   PLATFORM=""
fi
echo "Using platform $PLATFORM"

# Clean up old library build dirs and libraries for this platform
rm -rf library-build libraries${PLATFORM}

# Create new ones
mkdir -p libraries${PLATFORM}/include
mkdir -p libraries${PLATFORM}/lib

BUILDDIR=`mktemp -d "${TMPDIR:-/tmp}/vlasiator-library-build-XXXXX"`
ln -s $BUILDDIR library-build
cd library-build

# Build phiprof
git clone https://github.com/fmihpc/phiprof/
cd phiprof/src

if [[ $PLATFORM == "-arriesgado" ]]; then
   # Special workaround for missing include paths on arriesgado
   make -j 4 CCC=mpic++ CCFLAGS="-I /usr/lib/gcc/riscv64-linux-gnu/11/include -fpic -O2 -std=c++17 -DCLOCK_ID=CLOCK_MONOTONIC -fopenmp -W -Wall -Wextra -pedantic"
elif [[ $PLATFORM == "-appleM1" ]]; then
   make -j 4 CCC=mpic++ CC=appleLLVM CCFLAGS="-fpic -O2 -std=c++17 -DCLOCK_ID=CLOCK_MONOTONIC -fopenmp" LDFLAGS="-fopenmp"
elif [[ $PLATFORM == "-leonardo_dcgp_intel" ]]; then
   make -j 4 CCC="mpiicpc -cxx=icpx" CC="mpiicc -cc=iccx" CCFLAGS="-fpic -O2 -std=c++17 -DCLOCK_ID=CLOCK_MONOTONIC -qopenmp" LDFLAGS="-qopenmp"
else
   make -j 4 CCC=mpic++
fi
cp ../include/* $WORKSPACE/libraries${PLATFORM}/include
cp ../lib/* $WORKSPACE/libraries${PLATFORM}/lib
cd ../..

# Build VLSV
if [[ $PLATFORM != "-appleM1" ]]; then
   git clone https://github.com/fmihpc/vlsv.git
else
   git clone -b appleM1Build https://github.com/ursg/vlsv.git
fi
cd vlsv
if [[ $PLATFORM == "-leonardo_dcgp_intel" ]]; then
   make -j 4 CMP="mpiicpc -cxx=icpx"
else
   make -j 4
fi
cp libvlsv.a $WORKSPACE/libraries${PLATFORM}/lib
cp *.h $WORKSPACE/libraries${PLATFORM}/include
cd ..

# Build papi
if [[ $PLATFORM != "-arriesgado" && $PLATFORM != "-appleM1" ]]; then  # This fails on RISCV and MacOS
   git clone https://github.com/icl-utk-edu/papi
   cd papi/src
   if [[ $PLATFORM == "-leonardo_dcgp_intel" ]]; then
       # OneAPI compilers should use CC="mpiicc -cc=iccx" but this fails in configure
       ./configure --prefix=$WORKSPACE/libraries${PLATFORM} CC="mpiicc" CXX="mpiicpc -cxx=icpx"
   else
       ./configure --prefix=$WORKSPACE/libraries${PLATFORM} CC=mpicc CXX=mpic++
   fi
   make -j 4 && make install
   cd ../..
fi

# Build jemalloc
curl -O -L https://github.com/jemalloc/jemalloc/releases/download/5.3.0/jemalloc-5.3.0.tar.bz2
tar xjf jemalloc-5.3.0.tar.bz2
cd jemalloc-5.3.0
if [[ $PLATFORM == "-leonardo_dcgp_intel" ]]; then
    ./configure --prefix=$WORKSPACE/libraries${PLATFORM} --with-jemalloc-prefix=je_ CC="mpiicc" CXX="mpiicpc -cxx=icpx"
else
    ./configure --prefix=$WORKSPACE/libraries${PLATFORM} --with-jemalloc-prefix=je_ CC=mpicc CXX=mpic++
fi
make -j 4 && make install
cd ..

# Build Zoltan
git clone https://github.com/sandialabs/Zoltan.git
mkdir zoltan-build
cd zoltan-build
if [[ $PLATFORM == "-arriesgado" ]]; then
   ../Zoltan/configure --prefix=$WORKSPACE/libraries${PLATFORM} --enable-mpi --with-mpi-compilers --with-gnumake --with-id-type=ullong --host=riscv64-unknown-linux-gnu --build=arm-linux-gnu
elif [[ $PLATFORM == "-leonardo_dcgp_intel" ]]; then
   ../Zoltan/configure --prefix=$WORKSPACE/libraries${PLATFORM} --enable-mpi --with-mpi-compilers --with-gnumake --with-id-type=ullong CC="mpiicc" CXX="mpiicpc -cxx=icpx"
else
   ../Zoltan/configure --prefix=$WORKSPACE/libraries${PLATFORM} --enable-mpi --with-mpi-compilers --with-gnumake --with-id-type=ullong CC=mpicc CXX=mpic++
fi
make -j 4 && make install
cd ..

# Build boost
if [[ $PLATFORM == "-hile" || $PLATFORM == "-leonardo_booster" || $PLATFORM == "-leonardo_dcgp" || $PLATFORM == "-karolina_cuda" || $PLATFORM == "-karolina_gcc" ]]; then
    echo "### Downloading boost. ###"
    wget -q https://archives.boost.io/release/1.86.0/source/boost_1_86_0.tar.gz
    echo "### Extracting boost. ###"
    tar -xzf boost_1_86_0.tar.gz
    echo "### Building boost. ###"
    rm boost_1_86_0.tar.gz
    cd boost_1_86_0
    ./bootstrap.sh --with-libraries=program_options --prefix=$WORKSPACE/libraries${PLATFORM} stage
    ./b2
    echo "### Installing boost. ###"
    ./b2 install > /dev/null
    cd ..
fi

# Clean up build directory
rm -rf $BUILDDIR
