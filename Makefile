NVCC = nvcc
CXXFLAGS = -std=c++11
SRCS = kernel.cu main.cpp
TARGET = scs 

all: $(TARGET)

$(TARGET): $(SRCS)
	$(NVCC) $(CXXFLAGS) -o $@ $^

clean:
	rm -f $(TARGET)

