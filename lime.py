import numpy as np
import torch

from torchvision import transforms
from skimage.segmentation import slic

from utils_xai import normalize
import config_xai

from sklearn.metrics.pairwise import cosine_distances
from sklearn.linear_model import Ridge

class LIME:
    """
    Local Interpretable Model-agnostic Explanations (LIME)

    Ribeiro et al., 2016
    """

    def __init__(
        self,
        model,
        device,
        num_samples=1000,
        num_segments=150,
        compactness=10,
        sigma=0.25,
        batch_size=64,
    ):

        self.model = model
        self.device = device

        self.num_samples = num_samples
        self.num_segments = num_segments
        self.compactness = compactness
        self.sigma = sigma
        self.batch_size = batch_size

        self.mean = torch.tensor(
            config_xai.MEAN,
            device=device
        ).view(1,3,1,1)

        self.std = torch.tensor(
            config_xai.STD,
            device=device
        ).view(1,3,1,1)


    ####################################################################
    # Superpixel segmentation
    ####################################################################

    def _segment_image(self, image_original):
        """
        Segment image into superpixels using SLIC.

        Parameters
        ----------
        image_original : Tensor (3,H,W)

        Returns
        -------
        segments : ndarray (H,W)
        """

        image_np = image_original.permute(1, 2, 0).cpu().numpy()

        segments = slic(
            image_np,
            n_segments=self.num_segments,
            compactness=self.compactness,
            start_label=0
        )

        return segments


    ####################################################################
    # Random perturbations
    ####################################################################

    def _generate_masks(self, num_superpixels):
        """
        Generate binary masks.

        1 -> keep superpixel
        0 -> remove superpixel

        Returns
        -------
        ndarray (num_samples, num_superpixels)
        """

        masks = np.random.randint(
            0,
            2,
            size=(self.num_samples, num_superpixels),
            dtype=np.uint8
        )

        # first sample = original image
        masks[0] = 1

        return masks


    ####################################################################
    # Build perturbed images
    ####################################################################

    def _build_batch(
    self,
    image_original,
    segments,
    batch_masks,
    ):
        """
        Construct a batch of perturbed images.
        """

        batch_size = batch_masks.shape[0]

        images = image_original.unsqueeze(0).repeat(
            batch_size,
            1,
            1,
            1
        )

        # média RGB da imagem
        mean_color = image_original.mean(dim=(1, 2))

        # converter apenas uma vez
        segments = torch.from_numpy(segments).to(self.device)

        for sp in range(batch_masks.shape[1]):

            region = segments == sp

            remove = batch_masks[:, sp] == 0

            if remove.any():

                idx = np.where(remove)[0]

                for i in idx:

                    images[i][:, region] = mean_color.unsqueeze(1)

        images = images.to(self.device)

        images = (images - self.mean) / self.std

        return images

    def _compute_scores(
        self,
        image_original,
        segments,
        superpixel_masks,
    ):
        """
        Compute model predictions for all perturbed images.

        Parameters
        ----------
        image_original : Tensor (3,H,W)

        segments : ndarray (H,W)

        superpixel_masks : ndarray (num_samples, num_superpixels)

        Returns
        -------
        scores : ndarray (num_samples,)
            Model probability for each perturbation.
        """

        self.model.eval()

        scores = []

        with torch.no_grad():

            for start in range(0, self.num_samples, self.batch_size):

                end = min(start + self.batch_size, self.num_samples)

                batch_masks = superpixel_masks[start:end]

                images = self._build_batch(
                    image_original=image_original,
                    segments=segments,
                    batch_masks=batch_masks,
                )

                #images = images.to(self.device)

                outputs = self.model(images)

                probabilities = torch.sigmoid(outputs).squeeze(1)

                scores.append(probabilities.cpu())

        scores = torch.cat(scores, dim=0)

        return scores.numpy()

    def _compute_weights(self, superpixel_masks):
        """
        Compute proximity weights between each perturbed sample and the original image
        Parameters
        ----------
        superpixel_masks : ndarray (N,S)

        Returns
        -------
        weights : ndarray (N,)
        """
        original = np.ones((1, superpixel_masks.shape[1]))

        distances = cosine_distances(superpixel_masks, original).ravel()

        weights = np.exp( - (distances ** 2) / (self.sigma ** 2))

        return weights
    
    def _fit_surrogate(self, superpixel_masks, scores, weights):
        """
        Fit the local surrogate model.

        Parameters
        ----------
        superpixel_masks : ndarray (N,S)

        scores : ndarray (N,)

        weights : ndarray (N,)

        Returns
        -------
        coefficients : ndarray (S,)
        """

        model = Ridge(alpha=1.0)

        model.fit(superpixel_masks, scores, sample_weight=weights)

        return model.coef_
    
    def _build_heatmap(self, segments, coefficients):
        """
        Convert surrogate coefficients into a pixel-level heatmap
        Parameters
        ----------
        segments : ndarray (H,W)

        coefficients : ndarray (num_superpixels,)

        Returns
        -------
        Tensor (H,W)
        """

        heatmap = np.zeros_like(segments, dtype=np.float32)

        for sp, coef in enumerate(coefficients):
            heatmap[segments == sp] = coef
        
        heatmap = torch.from_numpy(heatmap)

        heatmap = normalize(heatmap)

        return heatmap

    ####################################################################
    # Main API
    ####################################################################

    def generate(
        self,
        image,
        image_original,
    ):
        """
        Generate LIME explanation.

        To be implemented.
        """

        # 1. Superpixel segmentation
        segments = self._segment_image(image_original)

        num_superpixels = np.unique(segments).size

        # 2. Generate perturbations
        superpixel_masks = self._generate_masks(num_superpixels)

        # 3. CNN predictions
        scores = self._compute_scores(
            image_original=image_original,
            segments=segments,
            superpixel_masks=superpixel_masks
        )

        # 4. Locality weights
        weights = self._compute_weights(superpixel_masks)

        # 5. Fit surrogate model
        coefficients = self._fit_surrogate(
            superpixel_masks,
            scores,
            weights
        )

        # 6. Build heatmap
        heatmap = self._build_heatmap(segments, coefficients)

        # Original prediction
        with torch.no_grad():
            output = self.model(image.unsqueeze(0).to(self.device))
            probability = torch.sigmoid(output)

        return {
            "heatmap": heatmap.cpu(),
            "logit": output.cpu(),
            "probability": probability.cpu(),
            "prediction": (probability >= 0.5).int().cpu()
        }