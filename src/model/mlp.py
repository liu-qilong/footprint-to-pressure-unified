import torch
from torch import nn

from src.tool.registry import MODEL_REGISTRY

@MODEL_REGISTRY.register()
class MLP(nn.Module):
    def __init__(self, device, img_width: int = 10):
        super().__init__()
        self.device = device
        self.img_width = img_width
        
        self.position_embedding = nn.Embedding(99, int(self.img_width * self.img_width / 2))
        self.young_embedding = nn.Linear(1, int(self.img_width * self.img_width / 2))

        self.model = nn.Sequential(
            nn.Linear(self.img_width * self.img_width * 2, 256),
            nn.Sigmoid(),
            nn.Linear(256, 512),
            nn.Sigmoid(),
            nn.Linear(512, 256),
            nn.Sigmoid(),
            nn.Linear(256, 128),
            nn.Sigmoid(),
            nn.Linear(128, 1),
        )

        # remember to send model to device!
        self.to(self.device)

    def forward(self, x):
        img_stack, young = x
        
        # reshape img_stack
        infer_shape = img_stack.shape[:-2] + (self.img_width * self.img_width,)  # e.g. (..., 198, 10, 10) -> (..., 198, 100)
        img_stack = img_stack.reshape(infer_shape)

        # positional embedding
        pos_arr = torch.concat([
            torch.arange(99, device=self.device),
            torch.arange(99, device=self.device),
            ], dim=-1,
        )
        pos_emb = self.position_embedding(pos_arr)
        pos_emb = pos_emb.expand(img_stack.shape[:-1] + (-1,))  # e.g. (198, 50) -> (..., 198, 50)

        # youngs' modulus embedding
        young_emb = self.young_embedding(young.unsqueeze(-1))
        young_emb = young_emb.unsqueeze(-2).expand(img_stack.shape[:-1] + (-1,))  # e.g. (50,) -> (..., 198, 50)

        x = torch.cat([img_stack, pos_emb, young_emb], dim=-1)
        
        return self.model(x).squeeze(-1)