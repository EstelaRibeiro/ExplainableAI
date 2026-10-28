# Definição dos caminhos:

OUTPUT_DIR = r'C:\Users\estela.ribeiro\JupyterNotebooks\estudando_cnns'
# Teste em Aorta Alongada
MODEL_NAMES = [
    "best_model_resnet50_fold_0_Aortic enlargement__crop_rees_all_1_q.pth",
    "best_model_resnet50_fold_1_Aortic enlargement__crop_rees_all_1_q.pth",
    "best_model_resnet50_fold_2_Aortic enlargement__crop_rees_all_1_q.pth",
    "best_model_resnet50_fold_3_Aortic enlargement__crop_rees_all_1_q.pth",
    "best_model_resnet50_fold_4_Aortic enlargement__crop_rees_all_1_q.pth",
]
MODEL_PATH = OUTPUT_DIR + '\models_aorta\\resnet50' # resnet50 / efficientnet_b4
MODEL_STR = 'resnet50' # resnet50 / efficientnet_b4

# Parametros:
RANDOM_SEED = 13
IMG_SIZE = (384, 384)
BATCH_SIZE = 8
LR = 1e-4
EPOCHS = 20
N_SPLITS = 5

# ImageNet normalization
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

DISPLAY_NAMES = {
    "saliency": "Saliency",
    "guided_backprop": "Guided BP",
    "smoothgrad_guided_backprop": "SmoothGrad\nGuided BP",
    "smoothgrad_saliency": "SmoothGrad\nSaliency",
    "integrated_gradients": "Integrated\nGradients",
    "gradcam": "Grad-CAM",
    "gradcampp": "Grad-CAM++",
    "layergradcam": "LayerCAM",
    "scorecam_otim": "ScoreCAM",
    "occlusion": "Occlusion",
    "lime": "LIME",
}