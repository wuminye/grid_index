#include "grid_index_gpu.cuh"

#define int64 int64_t

__global__
void grid_index(float* in_tensor, int x_dim, int y_dim, int z_dim, int feature_dim,
                 int64 *in_index, int num_points,
                 float* out_tensor )    // (N, dim)
{
    int ids = blockDim.x * blockIdx.x + threadIdx.x; //  index of point

    if (ids>=num_points) 
        return;

    int64* cur_index = in_index + ids*3;

    int64 xx = cur_index[0];
    int64 yy = cur_index[1];
    int64 zz = cur_index[2];

    if (xx <0 || xx>=x_dim || yy<0 || yy>=y_dim || zz<0 || zz>=z_dim )
        return;

    float *src_tensor = in_tensor + (xx*y_dim*z_dim + yy*z_dim + zz)*feature_dim;
    float *tar_tensor = out_tensor + ids*feature_dim;

    for (int i=0;i<feature_dim;++i)
    {
        tar_tensor[i] = src_tensor[i];
    }
      

}




void GPU_Grid_Index(torch::Tensor in_tensor, torch::Tensor in_index, torch::Tensor out_tensor)
{
    const auto num_points = in_index.size(0);

	dim3 dimBlock(256,1);
	dim3 dimGrid(num_points / dimBlock.x + 1, 1);

    grid_index << <dimGrid, dimBlock >> > (
		(float*)in_tensor.data<float>(), in_tensor.size(0),in_tensor.size(1),in_tensor.size(2),in_tensor.size(3),
		(int64*)in_index.data<int64>(),num_points,
        (float*)out_tensor.data<float>());
}



__global__
void grid_index_backward(float* grad_tensor,  // (N, dim)
                 int64 *in_index, int num_points,
                 float* out_grad_tensor, int x_dim, int y_dim, int z_dim, int feature_dim)   
{
    int ids = blockDim.x * blockIdx.x + threadIdx.x; //  index of point

    if (ids>=num_points) 
        return;

    int64* cur_index = in_index + ids*3;

    int64 xx = cur_index[0];
    int64 yy = cur_index[1];
    int64 zz = cur_index[2];

    if (xx <0 || xx>=x_dim || yy<0 || yy>=y_dim || zz<0 || zz>=z_dim )
        return;

    float *tar_tensor = out_grad_tensor + (xx*y_dim*z_dim + yy*z_dim + zz)*feature_dim;
    float *src_tensor = grad_tensor + ids*feature_dim;

    for (int i=0;i<feature_dim;++i)
        atomicAdd(tar_tensor + i, src_tensor[i]);

}


void GPU_Grid_Index_backward(
    torch::Tensor grad_tensor, 
    torch::Tensor in_index,
    torch::Tensor out_grad_tensor)
{
    const auto num_points = in_index.size(0);
    dim3 dimBlock(256,1);
	dim3 dimGrid(num_points / dimBlock.x + 1, 1);


    int x_dim = out_grad_tensor.size(0);
    int y_dim = out_grad_tensor.size(1);
    int z_dim = out_grad_tensor.size(2);
    int feature_dim = out_grad_tensor.size(3);


    grid_index_backward << <dimGrid, dimBlock >> > (
        (float*)grad_tensor.data<float>(),
		(int64*)in_index.data<int64>(),num_points,
        (float*)out_grad_tensor.data<float>(), x_dim,y_dim,z_dim,feature_dim);


}