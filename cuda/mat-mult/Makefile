NVCC    = nvcc
ARCH    = -arch=sm_52
CXX     = g++
OBJ     = main.o timer.o kernel.o
EXE     = matmult

default: $(EXE)

# Build CUDA source files (.cu)
%.o: %.cu
	$(NVCC) $(ARCH) -c -o $@ $<

# Build C++ source files (.cc)
%.o: %.cc
	$(CXX) -std=c++17 -c -o $@ $<

# Link with nvcc (important for CUDA runtime libs)
$(EXE): $(OBJ)
	$(NVCC) $(ARCH) $(OBJ) -o $(EXE)

clean:
	rm -rf $(OBJ) $(EXE)
