from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from sahi.models.base import DetectionModel
from sahi.prediction import ObjectPrediction
from sahi.utils.compatibility import fix_full_shape_list, fix_shift_amount_list


class RfdetrDetectionModel(DetectionModel):
    def __init__(
        self,
        model: Any | None = None,
        model_path: str | None = None,
        config_path: str | None = None,
        device: str | None = None,
        mask_threshold: float = 0.5,
        confidence_threshold: float = 0.3,
        category_mapping: dict | None = None,
        category_remapping: dict | None = None,
        load_at_init: bool = False,
        image_size: int | None = None,
    ):
        """Initialize the RfdetrDetectionModel with the given parameters.

        Args:
            model_path: str
                Path for the instance segmentation model weight
            config_path: str
                Path for the mmdetection instance segmentation model config file
            device: Torch device, "cpu", "mps", "cuda", "cuda:0", "cuda:1", etc.
            mask_threshold: float
                Value to threshold mask pixels, should be between 0 and 1
            confidence_threshold: float
                All predictions with score < confidence_threshold will be discarded
            category_mapping: dict: str to str
                Mapping from category id (str) to category name (str) e.g. {"1": "pedestrian"}
            category_remapping: dict: str to int
                Remap category ids based on category names, after performing inference e.g. {"car": 3}
            load_at_init: bool
                If True, automatically loads the model at initialization
            image_size: int
                Inference input size.
        """
        self._model = model
        self._device = device

        existing_packages = getattr(self, "required_packages", None) or []
        self.required_packages = [*list(existing_packages), "rfdetr"]

        super().__init__(
            model=model,
            model_path=model_path,
            config_path=config_path,
            device=device,
            mask_threshold=mask_threshold,
            confidence_threshold=confidence_threshold,
            category_mapping=category_mapping,
            category_remapping=category_remapping,
            load_at_init=True,
            image_size=image_size,
        ) 

    def set_model(self, model: Any, **kwargs):
        """
        This function should be implemented to instantiate a DetectionModel out of an already loaded model
        Args:
            model: Any
                Loaded model
        """
        self.model = model

    def _load_category_mapping_from_coco(self) -> dict | None:
        """
        Load category mapping from coco.json file in the same directory as model_path.

        Returns:
            dict mapping category id (int) to category name (str), or None if file not found
        """
        if not self.model_path:
            return None

        model_dir = Path(self.model_path).parent
        coco_path = model_dir / "coco.json"

        if not coco_path.exists():
            return None

        with open(coco_path, "r") as f:
            coco_data = json.load(f)

        if "categories" not in coco_data:
            return None

        category_mapping = {}
        for category in coco_data["categories"]:
            cat_id = category.get("id")
            cat_name = category.get("name")
            if cat_id is not None and cat_name is not None:
                category_mapping[cat_id] = cat_name

        return category_mapping if category_mapping else None

    def load_model(self):
        """This function should be implemented in a way that detection model should be initialized and set to
        self.model.

        (self.model_path, self.config_path, and self.device should be utilized)
        """
        # Load category mapping from coco.json if not already provided
        if not self.category_mapping:
            coco_mapping = self._load_category_mapping_from_coco()
            if coco_mapping:
                self.category_mapping = coco_mapping

        from rfdetr.detr import RFDETRBase, RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall

        model, model_path = self._model, self.model_path
        model_names = ("RFDETRBase", "RFDETRNano", "RFDETRSmall", "RFDETRMedium", "RFDETRLarge")
        if hasattr(model, "__name__") and model.__name__ in model_names:
            model_params = dict(
                resolution=int(self.image_size) if self.image_size else 560,
                device=self._device,
            )
            if self.category_mapping:
                model_params["num_classes"] = len(self.category_mapping.keys())
            if model_path:
                model_params["pretrain_weights"] = model_path

            model = model(**model_params)
        elif isinstance(model, (RFDETRBase, RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRLarge)):
            model = model
        else:
            raise ValueError(
                f"Model must be one of {model_names} models, got {self.model}."
            )
        self.set_model(model)
        # Reduce inference latency when supported (e.g. RF-DETR)
        if callable(getattr(self.model, "optimize_for_inference", None)):
            self.model.optimize_for_inference()

    def perform_inference(
        self,
        image: np.ndarray,
    ):
        """This function should be implemented in a way that prediction should be performed using self.model and the
        prediction result should be set to self._original_predictions.

        Args:
            image: np.ndarray
                A numpy array that contains the image to be predicted.
        """
        self._original_predictions = [self.model.predict(image, threshold=self.confidence_threshold)]

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list: list[list[int]] | None = [[0, 0]],
        full_shape_list: list[list[int]] | None = None,
    ):
        """This function should be implemented in a way that self._original_predictions should be converted to a list of
        prediction.ObjectPrediction and set to self._object_prediction_list.

        self.mask_threshold can also be utilized.
        Args:
            shift_amount_list: list of list
                To shift the box and mask predictions from sliced image to full sized image, should
                be in the form of List[[shift_x, shift_y],[shift_x, shift_y],...]
            full_shape_list: list of list
                Size of the full image after shifting, should be in the form of
                List[[height, width],[height, width],...]
        """
        # compatibility for sahi v0.8.15
        shift_amount_list = fix_shift_amount_list(shift_amount_list)
        full_shape_list = fix_full_shape_list(full_shape_list)

        object_prediction_list: list[ObjectPrediction] = []

        from supervision.detection.core import Detections

        original_detections: list[Detections] = self._original_predictions

        assert len(original_detections) == len(shift_amount_list) == len(full_shape_list), (
            "Length mismatch between original responses, shift amounts, and full shapes."
        )

        for original_detection, shift_amount, full_shape in zip(
            original_detections,
            shift_amount_list,
            full_shape_list,
        ):
            for xyxy, confidence, class_id in zip(
                original_detection.xyxy,
                original_detection.confidence,
                original_detection.class_id,
            ):
                if self.category_mapping:
                    category = self.category_mapping.get(int(class_id), None)
                else:
                    category = "Unknown"
                object_prediction = ObjectPrediction(
                    bbox=xyxy,
                    category_id=int(class_id),
                    category_name=category,
                    score=float(confidence),
                    shift_amount=shift_amount,
                    full_shape=full_shape,
                )
                object_prediction_list.append(object_prediction)

        object_prediction_list_per_image = [object_prediction_list]
        self._object_prediction_list_per_image = object_prediction_list_per_image
