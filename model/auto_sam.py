import torch
import torch.nn as nn
from segment_anything.modeling import Sam
from segment_anything.modeling.prompt_encoder import PositionEmbeddingRandom
from torch.nn import functional as F
from .prompt_embedding_generator import ResnetGenerator


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        y = self.weight[:, None, None] * x
        # y = torch.mul(self.weight[:, None, None], x)
        x = y + self.bias[:, None, None]
        return x


class AutoSam(nn.Module):
    def __init__(self, args, sam_model: Sam, mode='train'):
        super().__init__()
        self.args = args
        self.sam_model = sam_model
        self.prompt_embedding_generator = ResnetGenerator()
        self.position_embedding = PositionEmbeddingRandom(self.sam_model.prompt_encoder.embed_dim // 2)
        self.mode = mode

    def forward(self, x):

        for n, value in self.sam_model.named_parameters():
            if "Adapter" in n:
                value.requires_grad = True
            elif "output_hypernetworks_mlps" in n:
                value.requires_grad = True
            elif "output_upscaling" in n:
                value.requires_grad = True
            elif "iou_prediction_head" in n:
                value.requires_grad = True
            elif "neck" in n:
                value.requires_grad = True
            else:
                value.requires_grad = False

        x1, x2, x3, x4 = self.sam_model.image_encoder(x)

        image_embeddings = x4

        sparse_embeddings = torch.empty((image_embeddings.shape[0], 0, self.sam_model.prompt_encoder.embed_dim),
                                        device=x.device)
        dense_embeddings = self.prompt_embedding_generator(x)

        low_res_masks, iou_predictions = self.sam_model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=self.args.multimask,
        )

        masks = F.interpolate(low_res_masks, (self.args.image_size, self.args.image_size), mode="bilinear",
                              align_corners=False, )
        return masks


if __name__ == '__main__':
    x1 = torch.randn(4, 256, 64, 64)
    x2 = torch.randn(4, 256, 64, 64)
    x3 = torch.randn(4, 256, 64, 64)
