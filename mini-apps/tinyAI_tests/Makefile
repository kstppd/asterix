CC=nvcc
OPT=-O3
BIN=bin
CFLAGS=-std=c++20 -O3 --compiler-options "-g3 -ggdb -O3 -Wall -fopenmp"
INC_FLAGS=-I../../vdf_compression/cuda_mlp -I/home/kstppd/software/spdlog/include -Xcompiler=-isystem/home/kstppd/software/stb/ -I/usr/include/x86_64-linux-gnu/openblas-pthread/
LD_FLAGS=-L/usr/lib/x86_64-linux-gnu/openblas-pthread/ -lopenblas -lnvToolsExt -lcublas 
TORCH_INC_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=0 -I/home/kstppd/software/libtorch/include -I/home/kstppd/software/libtorch/include/torch/csrc/api/include
TORCH_LD_FLAGS=-L/home/kstppd/software/libtorch/lib/ -ltorch -ltorch_cpu  -lc10 

tiny: main_tiny.o
torch: main_torch.o 
mnist: mnist.o

train_torch.o:
	${CC} -c -x cu  train_torch.cpp ${CFLAGS} ${OPT} ${INC_FLAGS} ${TORCH_INC_FLAGS}

train_tiny.o:
	${CC} -c -x cu train_tiny.cpp ${CFLAGS} ${OPT} ${INC_FLAGS}

main_tiny.o: train_tiny.o
	${CC} image.cu train_tiny.o ${CFLAGS} ${OPT} ${INC_FLAGS} ${LD_FLAGS} -o ${BIN}

main_torch.o: train_torch.o
	${CC}  image.cu train_torch.o ${CFLAGS} ${OPT} ${INC_FLAGS} ${LD_FLAGS} ${TORCH_LD_FLAGS}  -o ${BIN}

mnist.o:
	${CC} mnist.cu ${CFLAGS} ${OPT} ${INC_FLAGS} ${LD_FLAGS} -lcurl -lz -o mnist_test

allclean:
	rm *o
	rm mnist_test
	rm ${BIN}
