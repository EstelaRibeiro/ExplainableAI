import os
import torch

from PIL import Image
from torchvision import transforms

import cv2
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np

# ImageNet defaults
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

def show_heatmap(heatmap):

    plt.figure(figsize=(6,6))
    plt.imshow(heatmap, cmap="jet", vmin=0, vmax=1)
    plt.axis("off")
    plt.show()

def get_image(img_dir, img_name, img_size):
    # Standard transformations for pre-trained models
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor()
    ])

    img_path = os.path.join(img_dir, img_name)
    image = Image.open(img_path).convert('RGB')

    image = transform(image)

    image_norm = transforms.Normalize(
        mean=MEAN,
        std=STD
    )(image.clone())

    return {
        "original": image,
        "normalized": image_norm
    }

def predict(image, model, device):
    model.eval()

    image = image.to(device)

    with torch.no_grad():
        output = model(image)
        prob = torch.sigmoid(output) # não converter para numpy aqui, porque alguns métodos XAI precisam dos tensores pra fazer calculos
        pred = (prob >= 0.5).int()

    return output, prob, pred

def normalize(x):
    x = x - x.min()
    x = x / (x.max() + 1e-8)
    return x

def tensor_to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()

    return x

def overlay_heatmap(
    original_img_path,
    heatmap,
    alpha=0.6,
    cmap=cv2.COLORMAP_TURBO,
    gamma=0.5,
    threshold=0.3,
):
    """
    Overlay a heatmap on the original image.

    Parameters
    ----------
    original_img_path : str
        Path to the original image.
    heatmap : torch.Tensor or np.ndarray
        Heatmap in range [0, 1].
    alpha : float
        Heatmap transparency.
    cmap : int
        OpenCV colormap.
    gamma : float
        Gamma correction (<1 increases contrast).
    threshold : float
        Values below this threshold are suppressed.
    """

    # Load original image
    image = np.array(Image.open(original_img_path).convert("RGB"))
    h, w = image.shape[:2]

    # Tensor -> numpy
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.detach().cpu().numpy()

    # Resize to original resolution
    heatmap = cv2.resize(
        heatmap,
        (w, h),
        interpolation=cv2.INTER_LINEAR,
    )

    # Normalize
    heatmap = heatmap.astype(np.float32)
    heatmap -= heatmap.min()
    heatmap /= (heatmap.max() + 1e-8)

    # Increase contrast
    heatmap = np.power(heatmap, gamma)

    # Remove weak activations
    #heatmap[heatmap < threshold] = 0.0

    # Renormalize
    heatmap /= (heatmap.max() + 1e-8)

    # Convert to uint8
    heatmap_uint8 = np.uint8(255 * heatmap)

    # Apply colormap
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cmap)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Overlay
    overlay = cv2.addWeighted(
        image,
        1 - alpha,
        heatmap_color,
        alpha,
        0,
    )

    return overlay

def show_explanation(original_img_path, heatmap, overlay, method_name='Saliency Map', show=False):
    """
    Display original image, heatmap and overlay side by side.

    Parameters
    ----------
    original_img_path : str
        Path to the original image.
    heatmap : torch.Tensor or np.ndarray
        Heatmap in [0, 1].
    overlay : np.ndarray
        Overlay image.
    """

    # Load original image
    image = np.array(Image.open(original_img_path).convert("RGB"))

    # Tensor -> numpy
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.detach().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image", fontsize=14)
    axes[0].axis("off")

    # Heatmap
    axes[1].imshow(heatmap, cmap="turbo", vmin=0, vmax=1)
    axes[1].set_title(method_name, fontsize=14)
    axes[1].axis("off")

    # Overlay
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay", fontsize=14)
    axes[2].axis("off")

    plt.tight_layout()
    if show:
        plt.show()
    
    return fig