import os
from pathlib import Path

from sahi import AutoDetectionModel
from huggingface_hub import hf_hub_download

from sdcat.logger import err


def create_model(model:str, conf:float, device:str, model_type=None):
    """
    Utility to determine the model type, model path, and create a detection model using SAHI.

    Args:
        model (str): The name of the model to use. Can be a predefined name or a local file path.
        conf (float): Confidence threshold for the model.
        device (str): The device to run the model on ('cpu', 'cuda', etc.).
        model_type (str): The type of model to use (e.g., 'yolov5', 'huggingface'). If None, will auto-detect

    Returns:
        detection_model: An instance of the AutoDetectionModel.
    """

    # Check if the provided model is a local file path
    if os.path.exists(model):
        if os.path.isdir(model):
            dir_to_model_map = { "yolov5": "yolov5", "yolov8": "yolov8", "huggingface": "huggingface" }
            model_path = [f for f in os.listdir(model) if f.endswith(".pt")]
            if len(model_path) == 0:
                err(f"No .pt file found in directory: {model}")
                raise ValueError(f"No .pt file found in directory: {model}")
            if model_type is None:
                for k, v in dir_to_model_map.items():
                    if k in model:
                        model_type = v
                        break
            if model_type is None:
                err(f"Could not determine model type from directory name: {model}. Try the --model-type option, e.g., --model-type yolov11")
                raise ValueError(f"Could not determine model type from directory name: {model}. Try the --model-type option, e.g., --model-type yolov11")
            detection_model = AutoDetectionModel.from_pretrained(
                model_type=model_type,
                model_path=Path(model) / model_path[0],
                confidence_threshold=conf,
                device=device,
            )
            return detection_model
        else:
            raise ValueError(f"Model path is not a directory: {model}")

    # Predefined model mapping
    model_map = {
        'yolov8s': {
            'model_type': 'yolov8',
            'model_path': 'ultralyticsplus/yolov8s'
        },
        'yolov8x': {
            'model_type': 'yolov8',
            'model_path': 'yolov8x.pt'
        },
        'hustvl/yolos-small': {
            'model_type': 'huggingface',
            'model_path': 'hustvl/yolos-small',
            'config_path': 'hustvl/yolos-small'
        },
        'hustvl/yolos-tiny': {
            'model_type': 'huggingface',
            'model_path': 'hustvl/yolos-tiny',
            'config_path': 'hustvl/yolos-tiny'
        },
        'MBARI-org/megamidwater': {
            'model_type': 'yolov5',
            'model_path': lambda: hf_hub_download("MBARI-org/megamidwater", "best.pt")
        },
        'MBARI-org/uav-yolov5-30k': {
            'model_type': 'yolov5',
            'model_path': lambda: hf_hub_download("MBARI-org/yolov5x6-uav-30k", "yolov5x6-uav-30k.pt")
        },
        'MBARI-org/uav-yolov5-18k': {
            'model_type': 'yolov5',
            'model_path': lambda: hf_hub_download("MBARI-org/yolov5-uav-18k", "yolov5x6-uav-18k.pt")
        },
        'MBARI-org/yolo11x-uavs-detect': {
            'model_type': 'yolo11',
            'model_path': lambda: hf_hub_download("MBARI-org/yolo11x-uavs-detect", "uavs-oneclass-best.pt")
        },
        'FathomNet/MBARI-315k-yolov5': {
            'model_type': 'yolov5',
            'model_path': lambda: hf_hub_download("FathomNet/MBARI-315k-yolov5", "mbari_315k_yolov5.pt")
        }
    }

    if model not in model_map:
        raise ValueError(f"Unknown model: {model}. Available models: {list(model_map.keys())}, "
                         f"or provide a local file path. You can also use the --model-type option to specify the model type.")

    model_info = model_map[model]
    model_type = model_info['model_type']
    model_path = model_info['model_path']

    if callable(model_path):  # If the path is a function (e.g., requires download)
        model_path = model_path()

    config_path = model_info.get('config_path', None)

    detection_model = AutoDetectionModel.from_pretrained(
        model_type=model_type,
        model_path=model_path,
        config_path=config_path,
        confidence_threshold=conf,
        device=device,
    )

    return detection_model