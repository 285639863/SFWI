all : SFWI2
CUDA_PATH = /usr/local/cuda
CC = mpicxx
NVCC = nvcc
CFLAGS = -O3 -w -fopenmp
NVINC = -I$(CUDA_PATH)/include
NVCCFLAGS = -O3 -w 
MACRO = -DELASTIC
LDFLAGS = -L$(CUDA_PATH)/lib64 -lcudart -lm -lcublas -lcufft

SFWI2 : FD2DGPU.o main.o born.o
	$(CC) -o $@ $^ $(NVINC) $(LDFLAGS) $(CFLAGS)

%.o : %.cu 
	$(NVCC)  $(NVINC) $(NVCCFLAGS) $(ORDER) $(MACRO) -o $@ -c $^  

%.o: %.cpp 
	$(CC)  $(NVINC) $(CFLAGS) $(MACRO) -o $@ -c $^

clean:
	rm -f main.o FD2DGPU.o born.o SFWI2

