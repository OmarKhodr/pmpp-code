NVCC    = nvcc
CXX     = g++
OBJ     = main.o timer.o
EXE     = vecadd

default: $(EXE)

# Build CUDA source files (.cu)
%.o: %.cu
	$(NVCC) -c -o $@ $<

# Build C++ source files (.cc)
%.o: %.cc
	$(CXX) -std=c++17 -c -o $@ $<

# Link with nvcc (important for CUDA runtime libs)
$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE)

clean:
	rm -rf $(OBJ) $(EXE)
