import torch
import torch.nn.functional as F

from utils_xai import normalize


class RISE:
    """
    RISE
    Randomized Input Sampling for Explanation

    Petsiuk et al., BMVC 2018
    """

    def __init__(
        self,
        model,
        device,
        num_masks=4000,
        grid_size=7,
        keep_prob=0.5,
        batch_size=64,
    ):

        self.model = model
        self.device = device

        self.num_masks = num_masks
        self.grid_size = grid_size
        self.keep_prob = keep_prob
        self.batch_size = batch_size

    def _generate_masks(self, image_size):

        H, W = image_size

        #
        # Random binary masks
        #
        masks = (
            torch.rand(
                self.num_masks,
                1,
                self.grid_size,
                self.grid_size,
                device=self.device,
            )
            < self.keep_prob
        ).float()

        #
        # Upsample
        #
        masks = F.interpolate(
            masks,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )

        return masks

    def generate(self, image):

        self.model.eval()

        image = image.unsqueeze(0).to(self.device)

        _, _, H, W = image.shape

        #
        # Original prediction
        #
        with torch.no_grad():
            output = self.model(image)

        probability = torch.sigmoid(output)

        #
        # Generate random masks
        #
        masks = self._generate_masks((H, W))

        #
        # Expand original image
        #
        images = image.expand(self.num_masks, -1, -1, -1)

        #
        # Apply masks
        #
        masked_images = images * masks

        #
        # Compute scores
        #
        scores = []

        with torch.no_grad():

            for i in range(0, self.num_masks, self.batch_size):

                batch = masked_images[i:i+self.batch_size]

                batch_scores = self.model(batch)

                scores.append(batch_scores.squeeze())

        scores = torch.cat(scores)

        #
        # Weighted average
        #
        heatmap = (
            scores.view(-1, 1, 1, 1) * masks
        ).sum(dim=0)

        #
        # Normalization from the paper
        #
        heatmap /= (self.num_masks * self.keep_prob)

        heatmap = normalize(heatmap.squeeze())

        return {
            "heatmap": heatmap.cpu(),
            "logit": output.detach().cpu(),
            "probability": probability.detach().cpu(),
            "prediction": (probability >= 0.5).int().cpu(),
        }