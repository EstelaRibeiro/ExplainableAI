import torch
import torch.nn.functional as F

from utils_xai import normalize

class LayerGradCAM:
    """
    Grad-CAM (selvaraju et al., 2017)
    """

    def __init__(self, model, device, target_layer=None):

        self.model = model
        self.device = device

        self.activations = None
        self.gradients = None

        self.target_layer = target_layer
        self.hooks_registered = False

        if self.target_layer is not None:
            print(f"GradCAM target layer: {self.target_layer}")
            self._register_hooks()

    def _find_target_layer(self, image):
        """
        Automatically finds the last layer that outputs
        a spatial feature map (B,C,H,W).
        """

        last_layer = None

        def hook(module, input, output):
            nonlocal last_layer

            if (
                isinstance(module, torch.nn.Conv2d)
                and isinstance(output, torch.Tensor)
                and output.ndim == 4
                and output.shape[-2] > 1
                and output.shape[-1] > 1
            ):
                last_layer = module

        handles = [
            module.register_forward_hook(hook)
            for module in self.model.modules()
        ]

        self.model.eval()

        with torch.no_grad():
            self.model(image)

        for h in handles:
            h.remove()

        if last_layer is None:
            raise RuntimeError("No suitable target layer found.")

        return last_layer

    def _register_hooks(self):

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

        self.hooks_registered = True

    def generate(self, image):

        self.model.eval()

        image = image.unsqueeze(0).to(self.device)

        #
        # Automatically find target layer only once
        #
        if self.target_layer is None:

            self.target_layer = self._find_target_layer(image)

            print(f"LayerCAM target layer: {self.target_layer}")

            self._register_hooks()

        self.model.zero_grad()

        output = self.model(image)

        #
        # Binary classification
        #
        score = output.squeeze()

        score.backward()

        activations = self.activations
        gradients = self.gradients

        ###############
        # AQUI A DIFERENÇA ENTRE GRADCAM

        # global average pooling
        weights = F.relu(gradients)

        # Weighted combination
        cam = (weights * self.activations).sum(dim=1)

        cam = F.relu(cam)

        # Upsampling
        cam = F.interpolate(
            cam.unsqueeze(1),
            size=image.shape[-2:],
            mode='bilinear',
            align_corners=False
        )

        cam = cam.squeeze()
        cam = normalize(cam)

        probability = torch.sigmoid(output)

        return {
            'heatmap': cam.cpu(),
            'logit': output.detach().cpu(),
            'probability': probability.detach().cpu(),
            'prediction': (probability >= 0.5).int().cpu()
        }
