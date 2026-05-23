import torch


def fn():
    torch.manual_seed(1337)
    _ = torch.randn(5)
    return torch.randn(5)


def sn():
    torch.manual_seed(1337)
    return torch.randn(5)


x = fn()
y = sn()
# torch.manual_seed(1337)
z = torch.randn(5)

print(x)
print(y)
print(z)
