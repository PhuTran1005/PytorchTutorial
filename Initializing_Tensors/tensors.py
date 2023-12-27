import torch

import numpy as np

#=============================================================================================#
#                                      Initializing Tensor                                    #
#=============================================================================================#

# set device to run tensors, prioritize the use of GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

my_tensor = torch.tensor([[1, 2, 3],
                        [4, 5, 6]], dtype=torch.float32, device=device, requires_grad=True)

print(my_tensor)
print(my_tensor.dtype)
print(my_tensor.device)
print(my_tensor.shape)
print(my_tensor.requires_grad)

# some common initialization methods
empty_tensor = torch.empty(size=(3, 3)) # empty tensor, not initialize
zero_tensor = torch.zeros((3, 3)) # zero tensor
rand_tensor = torch.rand((3, 3)) # random tensor
one_tensor = torch.ones((3, 3)) # one tensor
eye_tensor = torch.eye(3, 3) # diagonal tensor
arange_tensor = torch.arange(start=0, end=5, step=1) # a tensor from 0 to 5 with step is 1
linspace_tensor = torch.linspace(start=0.1, end=1, steps=10) 
empty_norm_tensor = torch.empty(size=(3, 3)).normal_(mean=0, std=1)
empty_uniform_tensor = torch.empty(size=(3, 3)).uniform_(0, 1)


# initialize and convert tensors to other types (int, float, double)
tensor = torch.arange(4)
print(tensor.bool()) # boolean True/False
print(tensor.short()) # int16
print(tensor.long()) # int64 (important)
print(tensor.half()) # float16
print(tensor.float()) # float32 (important)
print(tensor.double()) # float64

# array to tensor conversion and vice-versa
np_array = np.zeros((5, 5))
tensor = torch.from_numpy(np_array)
np_array_back = tensor.numpy()

print("numpy array", np_array)
print("numpy to torch tensor", tensor)
print("torch tensor to numpy")