import torch
import grid_index

import torch.nn as nn
class GIFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, in_tensor, in_index):

        num_points = in_index.size(0)
        dim_features = in_tensor.size(3)

        if in_index.size(1) != 3 or len(in_tensor.size())!=4:
            raise Exception('[Grid_Index] dimensions are not consistant.')



        out_tensor = torch.zeros(num_points,dim_features,device=in_tensor.device)

 
        grid_index.grid_index_forward(in_tensor,in_index,out_tensor)


        ctx.save_for_backward(in_tensor, in_index, out_tensor)

        return out_tensor

    @staticmethod
    def backward(ctx, grad_tensor):
        in_tensor, in_index, out_tensor = ctx.saved_tensors

        grad_tensor = grad_tensor.contiguous() 

        out_grad_tensor = torch.zeros_like(in_tensor)
        
        grid_index.grid_index_backward(grad_tensor,in_index,out_grad_tensor)

        return out_grad_tensor, None 



class Grid_Indexer(nn.Module):
    def __init__(self):
        super(Grid_Indexer, self).__init__()

    def forward(self, in_tensor, in_index):
        return GIFunction.apply(in_tensor, in_index)


indexer = Grid_Indexer()   

in_tensor = torch.ones(3,3,3,2).cuda()
in_index = torch.tensor([ [0,2,1],[0,2,1] ]).cuda().long()

out_tensor = torch.zeros(2,2).cuda()


ins = torch.nn.parameter.Parameter(in_tensor)


optimizer = torch.optim.Adam([ins], lr= 1e-1)



'''
res = indexer(ins,in_index)
loss = res.sum()
print(loss)
optimizer.zero_grad()
loss.backward()
print(ins.grad)
'''

for i in range(10):
    #res = torch.stack([ins[0,2,1,:], ins[0,2,1,:]])
    res = indexer(ins,in_index)
    loss = res.sum()
    print(loss)
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()
'''

def myf(ins,in_index):
    res = torch.stack([ins[0,2,1,:], ins[1,0,2,:]])
    return res




f = torch.autograd.gradcheck(myf, (ins,in_index), eps = 1e-6)
print(type(f))
'''