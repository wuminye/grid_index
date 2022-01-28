/*
@author:  Minye Wu
@contact: wuminye.x@gmail.com
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <torch/extension.h>


void GPU_Grid_Index(
    torch::Tensor in_tensor, 
    torch::Tensor in_index,
    torch::Tensor out_tensor);



void GPU_Grid_Index_backward(
    torch::Tensor grad_tensor, 
    torch::Tensor in_index,
    torch::Tensor out_grad_tensor);