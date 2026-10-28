import torch
from utils_xai import normalize

class Occlusion:
    """
    Zeiler & Ferguns (2014)
    """

    def __init__(self, model, device, window_size=32, stride=16, occlusion_value=0.0):
        self.model = model
        self.device = device
        self.window_size = window_size
        self.stride = stride
        self.occlusion_value = occlusion_value

    def generate(self, image):
        self.model.eval()

        image = image.unsqueeze(0).to(self.device)

        # Original prediction
        with torch.no_grad():
            output = self.model(image)
        
        original_score = output.squeeze()

        _, _, H, W = image.shape

        # Heatmap
        heatmap = torch.zeros((H,W), device = self.device)
        # Number of times each pixel was covered
        count_map = torch.zeros((H,W), device=self.device)

        # Sliding Window
        for y in range(0, H, self.stride):
            for x in range(0, W, self.stride):

                y2 = min(y + self.window_size, H)
                x2 = min(x + self.window_size, W)

                # Copy image
                occluded = image.clone()

                # Apply occlusion
                occluded[:, :, y:y2, x:x2] = self.occlusion_value

                # Forward
                with torch.no_grad():
                    score = self.model(occluded).squeeze()

                # Importance
                importance = original_score - score

                # Accumulate
                heatmap[y:y2, x:x2] += importance
                count_map[y:y2, x:x2] += 1

        # Average importance
        heatmap = heatmap / count_map.clamp(min=1)

        heatmap = normalize(heatmap)

        probability = torch.sigmoid(output)

        return {
            "heatmap": heatmap.cpu(),
            "logit": output.detach().cpu(),
            "probability": probability.detach().cpu(),
            "prediction": (probability >= 0.5).int().cpu(),
        }