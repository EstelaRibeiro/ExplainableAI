import torch
import torch.nn as nn

from utils_xai import normalize


# -------------------------------------------------------
# Guided ReLU (core do método)
# -------------------------------------------------------
class GuidedReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors

        grad_input = grad_output.clone()

        # Guided Backprop rule:
        grad_input[x <= 0] = 0
        grad_input[grad_output <= 0] = 0

        return grad_input


class GuidedReLU_Module(nn.Module):
    def forward(self, x):
        return GuidedReLU.apply(x)


# -------------------------------------------------------
# Replace ReLU in model
# -------------------------------------------------------
def replace_relu(module):
    for name, child in module.named_children():

        if isinstance(child, nn.ReLU):
            setattr(module, name, GuidedReLU_Module())

        else:
            replace_relu(child)


# -------------------------------------------------------
# Guided Backprop class
# -------------------------------------------------------
class GuidedBackprop:
    """Guided Backpropagation - Springenberg et al., 2014"""

    def __init__(self, model, device):
        self.model = model
        self.device = device

        self.model.eval()

        # IMPORTANT: replace ReLU ONCE
        replace_relu(self.model)

    def generate(self, image):
        """Returns Guided Backpropagation heatmap"""

        self.model.zero_grad()

        image = image.unsqueeze(0).to(self.device)
        image.requires_grad_(True)

        output = self.model(image)

        score = output.squeeze()
        score.backward()

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