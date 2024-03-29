import torch

#=============================================================================================#
#                                      Tensor indexing                                        #
#=============================================================================================#

batch_size = 10
features = 25
x = torch.rand((batch_size, features))

print(x)
print(x[0].shape) # return x[0, :]
print(x[:, 0]) # return 10
print(x[2, 0:10]) # 0:10 --> [0, 1, 2, ..., 9]

x[0, 0] = 100 # replacing

# fancy indexing
x = torch.arange(10)
indices = [2, 5, 8]
print(x[indices])

x = torch.rand((3, 5))
rows = torch.tensor([1, 0])
cols = torch.tensor([4, 0])
print(x[rows, cols].shape)

# more advanced indexing
x = torch.arange(10)
print(x[(x < 2) | (x > 8)]) # pick out 2 < x or x < 8

print(x[x.remainder(2) == 0])

# useful operations
print(torch.where(x > 5, x, x*2))
print(torch.tensor([0, 0, 1, 2, 3, 4]).unique())
print(x.ndimension()) # 5x5x5
print(x.numel()) # count the number of number in tensor