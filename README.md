# VectorAdd-PyTorch-cuda
This repository contains CUDA code for implementing vector addition on tensor. The mini-project was undertaken as an effort to strengthen my understanding of CUDA and Parallel Computing in general.

In this repository, we will build a simple CUDA based Vector Addition kernel. 




## Importing Headers and defining Macros
This code block includes necessary headers for CUDA programming with PyTorch. It also defines some macros and a utility function:

- CHECK_CUDA(x): A macro that checks if a given tensor x is a CUDA tensor.
- CHECK_CONTIGUOUS(x): A macro that checks if a given tensor x is contiguous.
- CHECK_INPUT(x): A macro that combines the above checks for CUDA and contiguity.
- The cdiv function is a utility function that performs ceiling division. It's used later in the code
```cuda
#include <torch/extension.h>
#include <stdio.h>
#include <c10/cuda/CUDAException.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

inline unsigned int cdiv(unsigned uint a, unsigned int b) { return (a + b - 1) / b; }
```
## Building a CUDA kernel

- VecAdd_kernel: This is the CUDA kernel responsible for performing vector addition on the GPU. It adds corresponding elements from the input array x and stores the result in the output array out.
- VecAdd: This is the PyTorch function that wraps the CUDA kernel. It checks the input tensor, calculates the size of the output tensor, and launches the CUDA kernel using the specified number of threads.

```cuda
// VecAdd_kernel: CUDA kernel for vector addition
__global__ void VecAdd_kernel(double* x, double* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = x[i] + x[i + n];
}

// VecAdd: PyTorch function for vector addition
torch::Tensor VecAdd(torch::Tensor input) {
    CHECK_INPUT(input);
    int h = input.size(0) / 2;
    auto output = torch::empty({h}, input.options());
    int threads = 256;
    VecAdd_kernel<<<cdiv(h, threads), threads>>>(
        input.data_ptr<double>(), output.data_ptr<double>(), h);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
```
## Creating 1D Vectors

- X and y: These are PyTorch tensors, each initialized with 10,000,000 random elements.
- combi: This tensor is created by concatenating the flattened versions of tensors X and y, resulting in a 1D tensor with a total of 20,000,000 elements.
- combi.contiguous().cuda(): This line ensures that the combi tensor is contiguous in memory and moves it to the CUDA device, making it suitable for GPU processing.

```python
# This code block initializes two 1D tensors, X and y, each containing 10,000,000 random elements.
X = torch.tensor(np.random.rand(10000000))
y = torch.tensor(np.random.rand(10000000))

# Concatenate X and y to create a 1D tensor, combi, with 20,000,000 elements in total.
combi = torch.cat((X.flatten(), y.flatten()))

# Create a contiguous 1D flattened CUDA tensor for kernel processing.
combi = combi.contiguous().cuda()
```
## CUDA Module Loading and Function Listing

- load_cuda: This function loads the CUDA module with the specified CUDA source code (cuda_src_mine), C++ source code (cpp_src), and a list of functions to be included in the module (in this case, ['VecAdd']).
- verbose=True: This flag enables verbose mode during the loading process, providing additional information about the compilation and loading steps.
- dir(module_mine): This command lists the attributes of the loaded CUDA module, allowing you to inspect the contents of the module, including the available functions.

```python
# Load the CUDA module with VecAdd function
module_mine = load_cuda(cuda_src_mine, cpp_src, ['VecAdd'], verbose=True)

# List the attributes of the loaded module
dir(module_mine)
```
## Executing the kernel

- module_mine.VecAdd_kernel(combi): This line executes the VecAdd_kernel function from the loaded CUDA module (module_mine) on the combi tensor.
- res: This variable stores the result of the kernel execution.

```cuda
res = module_mine.VecAdd_kernel(combi)
```


