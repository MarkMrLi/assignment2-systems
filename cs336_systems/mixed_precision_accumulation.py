import torch

s = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float32)
print(s)

s = torch.tensor(0, dtype=torch.float16)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float16)
print(s)

s = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float16)
print(s)

s = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    x = torch.tensor(0.01, dtype=torch.float16)
    s += x.type(torch.float32)
print(s)

s = torch.tensor(1.0, dtype=torch.float16)  # 累加到中途
x = torch.tensor(0.01, dtype=torch.float16)
print(f"x = {x}")
print(f"s = {s}")
print(f"s + 0.01 = {s + x}")
print(f"s == s + 0.01? {s == s + x}")