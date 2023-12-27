import torch

#=============================================================================================#
#                                      Tensor Reshaping                                       #
#=============================================================================================#

x = torch.arange(9)

x_3x3 = x.view(3, 3) # use for contiguous tensor, that mean tensor was stored congutiously in the memory
x_3x3 = x.reshape(3, 3)
print(x_3x3)

y = x_3x3.t() # transpose x_3x3
print(y)
# z = y.view(9) # error because the elements of tensor not contiguous
z = y.contiguous().view(9)
print(z)

x1 = torch.rand((2, 5))
x2 = torch.rand((2, 5))
print(torch.cat([x1, x2], dim=0).shape)
print(torch.cat([x1, x2], dim=1).shape)

z = x1.view(-1)
print(z.shape)

batch = 64
x = torch.rand((batch, 2, 5))
z = x.view(batch, -1)
print(z.shape)

z = x.permute(0, 2, 1)
print(z.shape)

x = torch.arange(10) # [10]
x = x.unsqueeze(0) # shape [1, 10]
print(x.shape)
z = x.unsqueeze(1) # shape [1, 1, 10]
print(z.shape)
z = x.unsqueeze(0).unsqueeze(1) # 1x1x1x10
print(z.shape)

z = x.squeeze(1)
print(z.shape)