import json
import os
from pathlib import Path

from sahi import AutoDetectionModel
from huggingface_hub import hf_hub_download
from transformers import AutoImageProcessor

from sdcat.detect.rfdetr_model import RfdetrDetectionModel
from sdcat.logger import err, info


def load_category_names_from_coco(coco_json_path: str) -> list:
    """
    Load category names from a COCO format JSON file.

    Args:
        coco_json_path: Path to the COCO JSON file

    Returns:
        list: A list of category names ordered by ID, e.g., ["cat", "dog"]
    """
    with open(coco_json_path, "r") as f:
        coco_data = json.load(f)

    categories = coco_data.get("categories", [])
    # Sort by ID to ensure correct indexing
    categories.sort(key=lambda x: x["id"])
    return [cat["name"] for cat in categories]


def find_coco_json(model_path: str) -> str | None:
    """
    Search for a coco.json file in the same directory as the model checkpoint.

    Args:
        model_path: Path to the model checkpoint file (.pth, .pt, etc.)

    Returns:
        str: Path to coco.json if found, None otherwise
    """
    model_dir = Path(model_path).parent
    coco_json_path = model_dir / "coco.json"
    if coco_json_path.exists():
        return str(coco_json_path)
    return None


def create_model(model: str, conf: float, device: str, model_type=None):
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
        if not os.path.isdir(model):
            raise ValueError(f"Model path is not a directory: {model}")

        pt_files = [f for f in os.listdir(model) if f.endswith((".pt", ".pth"))]
        if not pt_files:
            raise ValueError(f"No .pt or .pth file found in directory: {model}")

        if model_type is None:
            known_types = ["yolov5", "yolov8", "huggingface", "yolo11", "roboflow", "rfdetr"]
            model_type = next((t for t in known_types if t in model), None)

        if model_type is None:
            raise ValueError(
                f"Could not determine model type from directory name: {model}. "
                "Try the --model-type option, e.g., --model-type yolov11"
            )

        model_path = str(Path(model) / pt_files[0])

        # Search for coco.json in the same directory for category names
        category_names = None
        coco_json_path = find_coco_json(model_path)
        if coco_json_path:
            info(f"Found category names in {coco_json_path}")
            category_names = load_category_names_from_coco(coco_json_path)

        # Handle RF-DETR models with custom wrapper
        if model_type == "rfdetr":
            from rfdetr.detr import RFDETRLarge
            # Convert category_names list to category_mapping dict if provided
            category_mapping = None
            if category_names:
                category_mapping = {i: name for i, name in enumerate(category_names)}
            return RfdetrDetectionModel(
                model=RFDETRLarge,
                model_path=model_path,
                device=device,
                confidence_threshold=conf,
                category_mapping=category_mapping,
            )

        # Build kwargs for local models
        kwargs = {
            "model_type": model_type,
            "model_path": model_path,
            "confidence_threshold": conf,
            "device": device,
        }
        if category_names:
            kwargs["category_names"] = category_names

        # HuggingFace models require an image processor - load from local directory
        if model_type == "huggingface":
            image_processor = AutoImageProcessor.from_pretrained(model)
            kwargs["image_processor"] = image_processor

        return AutoDetectionModel.from_pretrained(**kwargs)

    # Predefined model mapping
    model_map = {
        "roboflow": {"model_type": "roboflow", "model": "RFDETRBase"},
        "yolov8s": {"model_type": "yolov8", "model_path": "ultralyticsplus/yolov8s"},
        "yolov8x": {"model_type": "yolov8", "model_path": "yolov8x.pt"},
        "hustvl/yolos-small": {
            "model_type": "huggingface",
            "model_path": "hustvl/yolos-small",
            "config_path": "hustvl/yolos-small",
            "image_processor": "hustvl/yolos-small",
        },
        "hustvl/yolos-tiny": {
            "model_type": "huggingface",
            "model_path": "hustvl/yolos-tiny",
            "config_path": "hustvl/yolos-tiny",
            "image_processor": "hustvl/yolos-tiny",
        },
        "MBARI-org/megamidwater": {
            "model_type": "yolov5",
            "model_path": lambda: hf_hub_download("MBARI-org/megamidwater", "best.pt"),
        },
        "MBARI-org/uav-yolov5-30k": {
            "model_type": "yolov5",
            "model_path": lambda: hf_hub_download("MBARI-org/yolov5x6-uav-30k", "yolov5x6-uav-30k.pt"),
        },
        "MBARI-org/rf-detrLarge-uavs-detectv0": {
            "model_type": "rfdetr",
            "rfdetr_model_class": "RFDETRLarge",
            "model_path": lambda: hf_hub_download("MBARI-org/rf-detrLarge-uavs-detectv0", "checkpoint_best_total.pth"),
        },
        "MBARI-org/uav-yolov5-18k": {
            "model_type": "yolov5",
            "model_path": lambda: hf_hub_download("MBARI-org/yolov5-uav-18k", "yolov5x6-uav-18k.pt"),
        },
        "MBARI-org/yolo11x-uavs-detect": {
            "model_type": "yolo11",
            "model_path": lambda: hf_hub_download("MBARI-org/yolo11x-uavs-detect", "uavs-oneclass-best.pt"),
        },
        "FathomNet/MBARI-315k-yolov5": {
            "model_type": "yolov5",
            "model_path": lambda: hf_hub_download("FathomNet/MBARI-315k-yolov5", "mbari_315k_yolov5.pt"),
        },
    }

    if model not in model_map:
        raise ValueError(
            f"Unknown model: {model}. Available models: {list(model_map.keys())}, "
            f"or provide a local file path. You can also use the --model-type option to specify the model type."
        )

    model_info = model_map[model]
    model_type = model_info["model_type"]
    model_path = model_info.get("model_path", None)

    if callable(model_path):  # If the path is a function (e.g., requires download)
        model_path = model_path()

    config_path = model_info.get("config_path", None)
    image_processor_name = model_info.get("image_processor", None)
 
    # Handle pretrained RF-DETR models with custom wrapper
    if model_type == "rfdetr":
        from rfdetr.detr import RFDETRBase, RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall
        rfdetr_model_class = model_info.get("rfdetr_model_class", "RFDETRLarge")
        model_classes = {
            "RFDETRBase": RFDETRBase,
            "RFDETRLarge": RFDETRLarge,
            "RFDETRMedium": RFDETRMedium,
            "RFDETRNano": RFDETRNano,
            "RFDETRSmall": RFDETRSmall,
        }
        model_class = model_classes.get(rfdetr_model_class, RFDETRLarge)
        model = model_class(pretrain_weights=model_path)
        detection_model = RfdetrDetectionModel(model=model, device=device, confidence_threshold=conf)
        return detection_model

    # Build kwargs for AutoDetectionModel (for other model types)
    kwargs = {
        "model_type": model_type,
        "confidence_threshold": conf,
        "device": device,
    }

    if model_path:
        kwargs["model_path"] = model_path
    if category_names:
        kwargs["category_names"] = category_names
    if config_path:
        kwargs["config_path"] = config_path
    if image_processor_name:
        # HuggingFace models require an image processor
        image_processor = AutoImageProcessor.from_pretrained(image_processor_name)
        kwargs["image_processor"] = image_processor

    detection_model = AutoDetectionModel.from_pretrained(**kwargs)

    return detection_model
