from sentence_transformers import SentenceTransformer

import numpy as np
import torch


class ClassTextEmbedding:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        return self.model.encode(texts)

from transformers import (
    ViTModel,
    AutoImageProcessor
)


class VisionTransformer:

    def __init__(
        self,
        model_name="facebook/dino-vits16",
        device="cuda"
    ):

        self.device = torch.device(
            device if torch.cuda.is_available()
            else "cpu"
        )

        self.processor = AutoImageProcessor.from_pretrained(
            model_name
        )

        self.model = ViTModel.from_pretrained(
            model_name
        ).to(self.device)

        self.model.eval()

    @torch.no_grad()
    def extract_features(self, images):

        inputs = self.processor(
            images=images,
            return_tensors="pt"
        )

        inputs = {
            k: v.to(self.device)
            for k, v in inputs.items()
        }

        outputs = self.model(**inputs)

        cls_token = outputs.last_hidden_state[:, 0]      # (B, 384,)
        patch_tokens = outputs.last_hidden_state[:, 1:]  # (B, 196,384)

        return (
            cls_token,
            patch_tokens
        )
        