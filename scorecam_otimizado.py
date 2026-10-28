import torch
import torch.nn.functional as F

from utils_xai import normalize


class ScoreCAMotim:
    """
    Score-CAM
    Wang et al., CVPR 2020

    https://arxiv.org/abs/1910.01279
    """

    def __init__(
        self,
        model,
        device,
        target_layer=None,
        batch_size=64,
        max_channels=None,
    ):

        self.model = model
        self.device = device

        self.target_layer = target_layer
        self.batch_size = batch_size
        self.max_channels = max_channels

        self.activations = None

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
        # Find target layer automatically
        #
        if self.target_layer is None:

            self.target_layer = self._find_target_layer(image)

            print(f"ScoreCAM target layer: {self.target_layer}")

            self._register_hooks()

        #
        # Forward pass
        #
        with torch.no_grad():
            output = self.model(image)

        activations = self.activations

        #
        # activations -> (1,C,H,W)
        #
        _, C, _, _ = activations.shape

        #
        # Optionally reduce the number of channels
        #
        if self.max_channels is not None:

            C = min(C, self.max_channels)

            activations = activations[:, :C]

        #
        # Upsample feature maps
        #
        activation_maps = F.interpolate(
            activations,
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        #
        # Normalize each activation map independently
        #
        mins = activation_maps.amin(dim=(2, 3), keepdim=True)
        maxs = activation_maps.amax(dim=(2, 3), keepdim=True)

        masks = (activation_maps - mins) / (maxs - mins + 1e-8)

        #
        # (1,C,H,W) -> (C,1,H,W)
        #
        masks = masks.squeeze(0).unsqueeze(1)

        #
        # Create masked images
        #
        images = image.expand(C, -1, -1, -1)

        masked_images = images * masks

        #
        # Compute Score-CAM weights
        #
        weights = []

        with torch.no_grad():

            for i in range(0, C, self.batch_size):

                batch = masked_images[i:i+self.batch_size]

                scores = self.model(batch)

                #
                # Binary classification
                #
                weights.append(scores.squeeze())

        weights = torch.cat(weights)

        #
        # Original paper:
        # weights = scores
        #
        # Optional normalization for numerical stability
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
            align_corners=False,
        )

        cam = normalize(cam.squeeze())

        probability = torch.sigmoid(output)

        return {
            "heatmap": cam.cpu(),
            "logit": output.detach().cpu(),
            "probability": probability.detach().cpu(),
            "prediction": (probability >= 0.5).int().cpu(),
        }