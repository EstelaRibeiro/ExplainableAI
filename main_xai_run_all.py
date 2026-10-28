import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import cv2
import math
import numpy as np

import config_xai
import utils_xai

from models_class import get_classification_model

# ===========================
# Métodos XAI
# ===========================
from saliency import Saliency
from guided_backprop import GuidedBackprop
from smoothgrad import SmoothGrad
from ig import IntegratedGradients

from gradcam import GradCAM
from gradcampp import GradCAMpp
from layergradcam import LayerGradCAM
from scorecam import ScoreCAM
from scorecam_otimizado import ScoreCAMotim
from lime import LIME

from occlusion import Occlusion


# ==========================================================
# DEVICE
# ==========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print(f"Device: {device}")
print(f"PyTorch: {torch.__version__}")

if torch.cuda.is_available():
    print(f"CUDA: {torch.version.cuda}")
    print(f"cuDNN: {torch.backends.cudnn.version()}")
    print(torch.cuda.get_device_name(0))
    torch.cuda.manual_seed(config_xai.RANDOM_SEED)
else:
    print("Running on CPU")


# ==========================================================
# LOAD MODEL
# ==========================================================
def load_model(model_name):

    path_model = os.path.join(
        config_xai.MODEL_PATH,
        model_name
    )

    model = get_classification_model(
        config_xai.MODEL_STR,
        pretrained=True
    )

    model.load_state_dict(
        torch.load(
            path_model,
            map_location=device
        )
    )

    model.to(device)
    model.eval()

    return model

# ==========================================================
# IMAGE
# ==========================================================
# AORTA
img_dir = r'C:\Users\estela.ribeiro\JupyterNotebooks\estudando_xai\aorta_alongada_images'
#img_name = '15f023463d88b8f8bc852a9a7b72e6e3_crop_rees_all_1_q' + '.png'
#img_name = 'fee7f5490aee6eb6626adb30d2509868_crop_rees_all_1_q' + '.png'
img_name = '0e8b19dc3af06aaed4718cd664a896bf_crop_rees_all_1_q' + '.png'

true_class = 1

sample = utils_xai.get_image(img_dir, 
                            img_name,
                            config_xai.IMG_SIZE)

image = sample['normalized']
image_original = sample['original']

image_path = os.path.join(img_dir, img_name)


# ==========================================================
# BUILD EXPLAINERS
# ==========================================================
def get_explainers(model, device):

    return {

        "saliency":
            Saliency(model, device),

        "guided_backprop":
            GuidedBackprop(model, device),

        "smoothgrad_guided_backprop":
            SmoothGrad(
                GuidedBackprop(model, device),
                device
            ),

        "integrated_gradients":
            IntegratedGradients(
                model,
                device
            ),

        "gradcam":
            GradCAM(
                model,
                device,
                target_layer=None
            ),

        "gradcampp":
            GradCAMpp(
                model,
                device,
                target_layer=None
            ),

        "layergradcam":
            LayerGradCAM(
                model,
                device,
                target_layer=None
            ),

        "scorecam_otim":
            ScoreCAMotim(
                model,
                device,
                target_layer=None
            ),

        "occlusion":
            Occlusion(
                model,
                device
            ),

        'lime':
            LIME(model, 
                 device)
    }


# ==========================================================
# RUN ONE METHOD
# ==========================================================
def run_method(method_name, explainer, results_dir):

    print("=" * 60)
    print(method_name)

    if method_name == 'lime':
        results = explainer.generate(image, image_original)
    else:
        results = explainer.generate(image)

    heatmap = results["heatmap"]

    print(f"Probability : {results['probability']}")
    print(f"Prediction  : {results['prediction']}")
    print(f"True class  : {true_class}")

    overlay = utils_xai.overlay_heatmap(
        image_path,
        heatmap,
        alpha=0.65,
        gamma=0.6,
        threshold=0.35,
    )

    fig = utils_xai.show_explanation(
        image_path,
        heatmap,
        overlay,
        method_name=method_name,
        show=False,
    )

    fig.savefig(
        os.path.join(results_dir, f"{method_name}.png"),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    return {
        "heatmap": heatmap,
        "overlay": overlay,
        "prediction": results["prediction"],
        "probability": results["probability"],
    }

# ==========================================================
# COMPARISON
# ==========================================================
def plot_comparison(
    comparison_results,
    save_path=None,
):
    """
    comparison_results deve possuir o formato:

    comparison_results = {

        "Fold 0": {
            "saliency": {...},
            "gradcam": {...},
            ...
        },

        "Fold 1": {
            ...
        }
    }
    """

    folds = list(comparison_results.keys())
    methods = list(comparison_results[folds[0]].keys())

    n_rows = len(methods)
    n_cols = len(folds)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols, 4 * n_rows)
    )

    # Caso exista apenas uma linha
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    # Caso exista apenas uma coluna
    if n_cols == 1:
        axes = np.expand_dims(axes, axis=1)

    for row, method in enumerate(methods):

        for col, fold in enumerate(folds):

            data = comparison_results[fold][method]

            overlay = data["overlay"]

            pred = data["prediction"]
            prob = data["probability"]

            if torch.is_tensor(pred):
                pred = pred.item()

            if torch.is_tensor(prob):
                prob = prob.item()

            ax = axes[row, col]

            ax.imshow(overlay)
            ax.axis("off")

            # Nome do FOLD apenas na primeira linha
            if row == 0:
                ax.set_title(
                    fold,
                    fontsize=14,
                    fontweight="bold"
                )

            # Nome do MÉTODO na primeira coluna
            if col == 0:
                ax.annotate(
                    config_xai.DISPLAY_NAMES.get(method, method),
                    xy=(-0.12, 0.5),
                    xycoords="axes fraction",
                    ha="right",
                    va="center",
                    fontsize=14,
                    fontweight="bold",
                    annotation_clip=False,
                )

            # Informação da predição
            ax.text(
                0.02,
                0.98,
                f"Prob={prob:.3f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=12,
                color="white",
                bbox=dict(
                    facecolor="black",
                    alpha=0.7,
                    pad=2
                )
            )

    #plt.tight_layout()
    plt.subplots_adjust(
        left=0.18,
        right=0.99,
        top=0.98,
        bottom=0.02,
        wspace=0.02,   # espaço horizontal entre colunas
        hspace=0.04    # espaço vertical entre linhas
    )

    if save_path is not None:
        plt.savefig(
            save_path,
            dpi=300,
            #bbox_inches="tight"
        )
        eps_path = os.path.splitext(save_path)[0] + ".eps"

        plt.savefig(
            eps_path,
            format="eps",
            #bbox_inches="tight"
        )

    plt.show()
    plt.close(fig)

# ==========================================================
# RUN ALL METHODS
# ==========================================================
comparison_results = {}
comparison_dir = "results"
os.makedirs(comparison_dir, exist_ok=True)

for fold, model_name in enumerate(config_xai.MODEL_NAMES):
    print('=' * 80)
    print(f'Loading Fold {fold}')

    results_dir = os.path.join("results", model_name, os.path.splitext(img_name)[0])
    os.makedirs(results_dir, exist_ok=True)

    model = load_model(model_name)

    explainers = get_explainers(model, device)    

    comparison_results[f'Fold {fold}'] = {}


    for method_name, explainer in explainers.items():

        try:

            result = run_method(
                method_name,
                explainer,
                results_dir
            )

            comparison_results[f'Fold {fold}'][method_name] = result

            print("-> Saved.")

        except Exception as e:

            print(f"-> Error running {method_name}")
            print(e)

plot_comparison(
    comparison_results=comparison_results,
    save_path=os.path.join(comparison_dir, f"{os.path.splitext(img_name)[0]}_comparison.png")
)

print("=" * 60)
print("Finished.")