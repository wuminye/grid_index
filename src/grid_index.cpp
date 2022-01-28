#include <torch/extension.h>
#include <vector>
#include "grid_index_gpu.cuh"
#include <iostream>



// CUDA forward declarations

std::vector<torch::Tensor> grid_index_forward(
    torch::Tensor in_tensor, //(X,Y,Z,dim)
    torch::Tensor in_index,  // (N,3)
    torch::Tensor out_tensor // (N, dim)
    );


std::vector<torch::Tensor> grid_index_backward(
    torch::Tensor grad_tensor, //(N, dim)
    torch::Tensor in_index,     //(N,3)
    torch::Tensor out_grad_tensor // (X,Y,Z, dim)
    );

#define CHECK_CPU(x) AT_ASSERTM(!x.is_cuda(), #x " must be a CPU tensor")
#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) AT_ASSERTM(x.scalar_type()==torch::ScalarType::Float, #x " must be a float tensor")
#define CHECK_INT(x) AT_ASSERTM(x.scalar_type()==torch::ScalarType::Int, #x " must be a Int tensor")
#define CHECK_SHORT(x) AT_ASSERTM(x.scalar_type()==torch::ScalarType::Short, #x " must be a Int tensor")
#define CHECK_LONG(x) AT_ASSERTM(x.scalar_type()==torch::ScalarType::Long, #x " must be a Long tensor")
#define CHECK_UCHAR(x) AT_ASSERTM(x.scalar_type()==torch::ScalarType::Byte, #x " must be a Int tensor")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x);

std::vector<torch::Tensor> grid_index_forward(
    torch::Tensor in_tensor, //(X,Y,Z,dim)
    torch::Tensor in_index,  // (N,3)
    torch::Tensor out_tensor // (N, dim)
    )
{
    CHECK_INPUT(in_tensor); CHECK_FLOAT(in_tensor);
    CHECK_INPUT(in_index); CHECK_LONG(in_index);
    CHECK_INPUT(out_tensor); CHECK_FLOAT(out_tensor);

    AT_ASSERTM(in_tensor.size(3)== out_tensor.size(1), "in_tensor and out_tensor must be the same size in feature dim");
    AT_ASSERTM(in_index.size(0)== out_tensor.size(0), "in_index and out_tensor must be the same number N");
    AT_ASSERTM(in_index.size(1)== 3, "in_index must be (N,3)");

    GPU_Grid_Index(
        in_tensor, 
        in_index,
        out_tensor);
    
    return {out_tensor};
}


std::vector<torch::Tensor> grid_index_backward(
    torch::Tensor grad_tensor, //(N, dim)
    torch::Tensor in_index,     //(N,3)
    torch::Tensor out_grad_tensor // (X,Y,Z, dim)
    )
{
    CHECK_INPUT(grad_tensor); CHECK_FLOAT(grad_tensor);
    CHECK_INPUT(in_index); CHECK_LONG(in_index);
    CHECK_INPUT(out_grad_tensor); CHECK_FLOAT(out_grad_tensor);

    AT_ASSERTM(grad_tensor.size(0)== in_index.size(0), "in_index and grad_tensor must be the same number N");
    AT_ASSERTM(grad_tensor.size(1)== out_grad_tensor.size(3), "grad_tensor and out_grad_tensor must be the same size in feature dim");
    AT_ASSERTM(in_index.size(1)== 3, "in_index must be (N,3)");

    GPU_Grid_Index_backward(
        grad_tensor, 
        in_index,
        out_grad_tensor);

    return {out_grad_tensor};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("grid_index_forward", &grid_index_forward, "grid_index forward (CUDA)");
  m.def("grid_index_backward", &grid_index_backward, "grid_index backward (CUDA)");
}