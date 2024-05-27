# Makefile for compiling CUDA program

# Compiler settings
NVCC=nvcc

# NVCC flags
# Pass -O3 to the host compiler, specify the compute architecture
NVCCFLAGS=-Xcompiler -O3 -arch=sm_70

# Linker flags
LDFLAGS=-lcublas

# Target binary
TARGET=rawcuda

# Source file
SOURCE=rawcuda.cu

# Default rule
all: $(TARGET)

$(TARGET): $(SOURCE)
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(LDFLAGS)

# Clean rule
clean:
	rm -f $(TARGET)

.PHONY: all clean
