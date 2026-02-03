from torch import nn
import torch
from cs336_basics.nn_utils import cross_entropy
class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features,10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x:torch.Tensor):
        x = self.fc1(x)
        print(f"The output's dtype of the first feed-forward layer:{x.dtype}")
        x = self.relu(x)
        print(f"The dtype after relu:{x.dtype}")
        x = self.ln(x)
        print(f"The dtype after ln:{x.dtype}")
        x = self.fc2(x)
        print(f"The dtype of model's predicted logits:{x.dtype}")
        return x
    
def main():
    device = 'cuda' if torch.cuda.is_available()  else 'cpu'
    model = ToyModel(100,100).to(device)
    x = torch.ones([1, 100],device=device)
    y = torch.tensor([10], dtype=int, device=device)

    print(f"The model parameters out of autocast context:{model.fc1.weight.dtype}")
    with torch.autocast(device_type=device,dtype=torch.bfloat16):
        print(f"The model parameters within autocast context:{model.fc1.weight.dtype}")
        y_hat = model.forward(x)

        loss = cross_entropy(y_hat, y)
        print(f"The dtype of loss:{loss.dtype}")

        loss.backward()
        print(f"The dtype of model's gradients:{model.fc1.weight.grad.dtype}")


if __name__ == "__main__":
    main()
