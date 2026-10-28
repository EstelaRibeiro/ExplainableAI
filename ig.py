import torch

from utils_xai import normalize


class IntegratedGradients:
    """
    Integrated Gradients
    Sundararajan et al. (2017)
    """

    def __init__(self, model, device, steps=50):
        self.model = model
        self.device = device
        self.steps = steps

    def generate(self, image, baseline=None):

        self.model.eval()

        image = image.unsqueeze(0).to(self.device)

        #
        # Default baseline = black image
        #
        if baseline is None:
            baseline = torch.zeros_like(image)
        else:
            baseline = baseline.unsqueeze(0).to(self.device)

        #
        # Difference between image and baseline
        #
        delta = image - baseline

        #
        # Accumulate gradients
        #
        accumulated_gradients = torch.zeros_like(image)

        #
        # Interpolation coefficients
        #
        alphas = torch.linspace(
            0,
            1,
            self.steps,
            device=self.device
        )

        output = None

        for alpha in alphas:

            #
            # Interpolated image
            #
            interpolated = baseline + alpha * delta

            interpolated.requires_grad_(True)

            self.model.zero_grad()

            output = self.model(interpolated)

            score = output.squeeze()

            score.backward()

            accumulated_gradients += interpolated.grad.detach()

        #
        # Average gradient
        #
        avg_gradients = accumulated_gradients / self.steps

        #
        # Integrated gradients
        #
        integrated_gradients = delta * avg_gradients

        #
        # RGB -> grayscale heatmap
        #
        heatmap, _ = integrated_gradients.abs().max(dim=1)

        heatmap = normalize(heatmap.squeeze())

        probability = torch.sigmoid(output)

        return {
            "heatmap": heatmap.cpu(),
            "logit": output.detach().cpu(),
            "probability": probability.detach().cpu(),
            "prediction": (probability >= 0.5).int().cpu()
        }