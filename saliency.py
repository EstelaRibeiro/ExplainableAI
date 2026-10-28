import torch
from utils_xai import normalize

class Saliency:
    """Vanilla Saliency Maps - Simonyan et al., 2013"""

    def __init__(self, model, device):
        self.model = model
        self.device = device

    def generate(self, image):
        self.model.eval()

        image = image.unsqueeze(0).to(self.device)

        # queremos guardar as derivadas da imagem, ou seja, como a saída muda quando altero cada pixel?
        image.requires_grad_(True)

        if image.grad is not None:
            image.grad.zero_()

        # limpar os gradientes
        self.model.zero_grad()

        output = self.model(image)
        
        output.backward()

        gradient = image.grad.detach()

        heatmap, _ = gradient.abs().max(dim=1)

        heatmap = heatmap.squeeze()

        heatmap = normalize(heatmap)

        return {
                "heatmap": heatmap.cpu(),
                "logit": output.detach().cpu(),
                "probability": torch.sigmoid(output).detach().cpu(),
                "prediction": (torch.sigmoid(output) >= 0.5).int().cpu()
                }
    