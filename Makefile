DEBUG=-g
OPT=-O2
OBJ = scan.o

all: fill-debug fill-opt

fill-debug: fill.cu scan.cu
	nvcc $(DEBUG) -o $@ fill.cu scan.cu

fill-opt: fill.cu scan.cu
	nvcc $(DEBUG) $(OPT) -o $@ fill.cu scan.cu

clean:
	rm -rf fill-*
