import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import numpy as np

# Parameters for the simulation
# TODO: add parameters as needed

# Variables for the simulation
# TODO: add variables as needed
# TODO: remove example variables below
a = np.random.randn(4,4) # generate some data
a = a.astype(np.float32) # convert to float32
a_gpu = cuda.mem_alloc(a.nbytes) # allocate GPU memory
cuda.memcpy_htod(a_gpu, a) # transfer data to GPU

# Kernels for the simulation
# TODO: add kernels as needed
# TODO: remove example kernel below
mod = SourceModule("""
__global__ void doublify(float *a) {
  int idx = threadIdx.x + threadIdx.y*4;
  a[idx] *= 2;
}
""")


# Main simulation loop
# TODO: add main simulation loop as needed
# TODO: remove example code below
func = mod.get_function("doublify")
func(a_gpu, block=(4,4,1)) # launch kernel

a_doubled = np.empty_like(a) # create empty array for result
cuda.memcpy_dtoh(a_doubled, a_gpu) # transfer data back to host
print("Original array:")
print(a)
print("Doubled array:")
print(a_doubled)

