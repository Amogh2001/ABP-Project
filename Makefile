NVCC = nvcc
CXX = /usr/bin/g++
CUDA_PATH = /mnt/116bae52-4a8e-4b7e-a3f7-857a4ec210ed/cuda
CXXFLAGS = -std=c++11
NVCCFLAGS = -rdc=true
LIBS = -L$(CUDA_PATH)/lib64 -lcusparse
SRCS = kernel.cu main.cu 
TARGET = scs

all: $(TARGET)

$(TARGET): $(SRCS)
        $(NVCC) -ccbin $(CXX) $(CXXFLAGS) $(NVCCFLAGS) -o $@ $^ $(LIBS)

clean:
        rm -f $(TARGET)
