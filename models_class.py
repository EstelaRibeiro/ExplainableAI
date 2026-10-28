# import torch.nn as nn
import timm 
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_classification_model(model_name: str, pretrained: bool = True, num_classes: int = 1):
    """
    Carrega um modelo pré-treinado do torchvision e adapta a camada final para classificação.
    """
    try:
        # Usar timm.create_model, que é a forma padrão e mais poderosa para
        # instanciar modelos nesta biblioteca.
        model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes  # timm substitui o "head" do classificador automaticamente.
        )
        print(f"Modelo '{model_name}' criado com sucesso via timm.")
        return model.to(device)
    except Exception as e:
        print(f"Erro ao criar o modelo '{model_name}' com timm. Verifique se o nome está correto.")
        print(f"Erro original: {e}")
        raise  # Lança a exceção para interromper a execução se o modelo não for encontrado.