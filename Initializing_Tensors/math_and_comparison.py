import torch

#=============================================================================================#
#                               Tensor Math & Comparison Operations                           #
#=============================================================================================#

x = torch.tensor([1, 2, 3])
y = torch.tensor([9, 8, 7])

# addition
z1 = torch.empty(3) # method 1
torch.add(x, y, out=z1)

z2 = torch.add(x, y) # method 2
z3 = x + y # method 3 (preferable)

print(z1)
print(z2)
print(z3)

# subtraction
z = x - y

print(z)

# division
z = torch.true_divide(x, y) # ele-wise operation

# inplace operations
t = torch.zeros(3)
t.add_(x)
t += x

# exponentiation
z1 = x.pow(2) # method 1
z2 = x ** 2 # method 2

# simple comparison
z = x > 0
z = x < 0

# matrix maniplication
x1 = torch.rand([2, 5])
x2 = torch.rand([5, 3])

x3 = torch.mm(x1, x2) # method 1, output shape: 2x3
x3 = x1.mm(x2) # method 2

# matrix exponentiation
matrix_exp = torch.rand(5, 5)

print(matrix_exp.matrix_power(3))

# element wise mult
z = x * y

print(z)

# dot product
z = torch.dot(x, y)

print(z)

# batch matrix multiplication
batch = 32
n = 10
m = 20
p = 30

tensor1 = torch.randn((batch, n, m))
tensor2 = torch.rand((batch, m, p))
out_bmm = torch.bmm(tensor1, tensor2) # shape (batch, n, p)

# example of broadcasting
x1 = torch.randn((5, 5))
x2 = torch.randn((1, 5))

z = x1 - x2
z = x1 ** x2

# other useful tebsor operations
sum_x = torch.sum(x, dim=0)
values, indices = torch.max(x, dim=0)
values, indices = torch.min(x, dim=0)
abs_x = torch.abs(x)
z = torch.argmax(x, dim=0)
z = torch.argmin(x, dim=0)
mean_x = torch.mean(x.float(), dim=0) # convert x to float because mean() need float variable
z = torch.eq(x, y)
sorted_y, indices = torch.sort(y, dim=0, descending=False)

z = torch.clamp(x, min=0) # check tensor x, and if have any elements < 0 will be converted to 0

x = torch.tensor([1, 0, 1, 1, 1], dtype=torch.bool)
z = torch.any(x) # if have any elements True, return True otherwise return False
z = torch.all(x) # if have any elements False, return False otherwise return Trie