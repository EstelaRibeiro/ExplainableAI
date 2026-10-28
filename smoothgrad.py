import torch

from utils_xai import normalize


class SmoothGrad:
    """
    SmoothGrad for XAI methods.

    Works with:
    - Saliency
    - Guided Backprop
    """

    def __init__(self, explainer, device, num_samples=30, noise_level=0.15):
        """
        Parameters
        ----------
        explainer : object
            Must implement generate(image)
        device : torch.device
        num_samples : int
            Number of noisy samples
        noise_level : float
            Std of Gaussian noise (relative to image range)
        """

        self.explainer = explainer
        self.device = device
        self.num_samples = num_samples
        self.noise_level = noise_level

    def generate(self, image):
        """
        Returns SmoothGrad heatmap
        """

        image = image.to(self.device)

        all_maps = []

        original_result = self.explainer.generate(image)

        for _ in range(self.num_samples):

            # --------------------------------------------------
            # add gaussian noise
            # --------------------------------------------------
            noise = torch.randn_like(image) * self.noise_level
            noisy_image = image + noise

            # --------------------------------------------------
            # compute explanation
            # --------------------------------------------------
            result = self.explainer.generate(noisy_image)

            heatmap = result["heatmap"] if isinstance(result, dict) else result

            all_maps.append(heatmap)

        # --------------------------------------------------
        # average maps
        # --------------------------------------------------
        smooth_map = torch.stack(all_maps).mean(dim=0)

        smooth_map = normalize(smooth_map)

        return {
            "heatmap": smooth_map.cpu(),
            "logit": original_result["logit"],
            "probability": original_result["probability"],
            "prediction": original_result["prediction"],
        }