import torch
import torchvision
import numpy as np
from typing import List, Dict, Any

class VehicleDetector:
    def __init__(self, model_path: str = None):
        """
        Initialize the vehicle detector
        Args:
            model_path: Path to the PyTorch model
        """
        # Load the pre-trained model
        if model_path:
            self.model = torch.load(model_path)
        else:
            # Use a pre-trained model from torchvision
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        
        self.model.eval()  # Set to evaluation mode
        
        # COCO dataset class names
        self.class_names = ['N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                           'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                           'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                           'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                           'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                           'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                           'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                           'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                           'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                           'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
                           'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
                           'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

        # Vehicle classes we're interested in
        self.vehicle_classes = ['car', 'truck', 'bus', 'motorcycle', 'ambulance']

    def detect(self, frame: np.ndarray, confidence_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Detect vehicles in the frame
        Args:
            frame: Input frame
            confidence_threshold: Minimum confidence score for detection
        Returns:
            List of detections
        """
        # Convert frame to tensor
        # PyTorch expects channels first format (C, H, W)
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float()
        
        # Add batch dimension
        frame_tensor = frame_tensor.unsqueeze(0)
        
        # Run detection
        with torch.no_grad():
            detections = self.model(frame_tensor)
        
        # Process detections
        boxes = detections[0]['boxes'].numpy()
        scores = detections[0]['scores'].numpy()
        labels = detections[0]['labels'].numpy()

        results = []
        for i in range(len(scores)):
            if scores[i] >= confidence_threshold:
                class_name = self.class_names[labels[i]]
                if class_name in self.vehicle_classes:
                    # Convert from [x1, y1, x2, y2] to [y1, x1, y2, x2] format
                    # to match the original TensorFlow format
                    bbox = boxes[i]
                    results.append({
                        'bbox': [bbox[1], bbox[0], bbox[3], bbox[2]],
                        'label': class_name,
                        'confidence': scores[i]
                    })

        return results

    def count_vehicles(self, detections: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Count vehicles by type
        Args:
            detections: List of detections
        Returns:
            Dictionary with vehicle counts
        """
        counts = {vehicle: 0 for vehicle in self.vehicle_classes}
        for detection in detections:
            counts[detection['label']] += 1
        return counts 