import torch
import torch.nn.functional as F

from utils_xai import normalize


class ScoreCAM:
    """
    Score-CAM
    Wang et al., CVPR 2020
    """

    def __init__(self,
                 model,
                 device,
                 target_layer=None,
                 max_channels=None):

        self.model = model
        self.device = device

        self.activations = None

        self.target_layer = target_layer
        self.max_channels = max_channels

        if self.target_layer is not None:
            print(f"ScoreCAM target layer: {self.target_layer}")
            self._register_hooks()

    def _find_target_layer(self, image):
        """
        Automatically finds the last layer producing
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

        self.target_layer.register_forward_hook(forward_hook)

    def generate(self, image):

        self.model.eval()

        image = image.unsqueeze(0).to(self.device)

        #
        # Automatically determine target layer
        #
        if self.target_layer is None:

            self.target_layer = self._find_target_layer(image)

            print(f"ScoreCAM target layer: {self.target_layer}")

            self._register_hooks()

        #
        # Initial forward
        #
        with torch.no_grad():
            output = self.model(image)

        activations = self.activations

        B, C, H, W = activations.shape

        #
        # Limit number of channels (optional)
        #
        if self.max_channels is not None:
            C = min(C, self.max_channels)
            activations = activations[:, :C]

        #
        # Upsample feature maps
        #
        upsampled = F.interpolate(
            activations,
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        weights = []

        #
        # Compute one score for each feature map
        #
        for c in range(C):

            mask = upsampled[:, c:c+1]

            #
            # Normalize mask to [0,1]
            #
            mask = mask - mask.min()

            if mask.max() > 0:
                mask = mask / mask.max()

            #
            # Masked image
            #
            masked_image = image * mask

            with torch.no_grad():
                score = self.model(masked_image)

            #
            # Binary classification:
            # use the logit before sigmoid
            #
            weights.append(score.squeeze())

        weights = torch.stack(weights)

        #
        # Normalize weights
        #
        weights = F.softmax(weights, dim=0)

        #
        # Weighted combination
        #
        cam = (
            weights.view(1, C, 1, 1)
            * activations
        ).sum(dim=1)

        cam = F.relu(cam)

        #
        # Upsample CAM
        #
        cam = F.interpolate(
            cam.unsqueeze(1),
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        cam = normalize(cam.squeeze())

        probability = torch.sigmoid(output)

        return {
            "heatmap": cam.cpu(),
            "logit": output.detach().cpu(),
            "probability": probability.detach().cpu(),
            "prediction": (probability >= 0.5).int().cpu()
        }