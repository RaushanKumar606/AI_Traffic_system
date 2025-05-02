import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any
from performance_optimizer import PerformanceOptimizer

class VideoProcessor:
    def __init__(self, source: int = 0, target_fps: int = 30):
        """
        Initialize the video processor
        Args:
            source: Camera index or video file path
            target_fps: Target frames per second
        """
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise ValueError("Failed to open video source")
        
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        # Initialize performance optimizer
        self.optimizer = PerformanceOptimizer(target_fps=target_fps)
        
        # Get optimized frame size
        self.optimized_width, self.optimized_height = self.optimizer.get_optimized_frame_size(
            (self.frame_width, self.frame_height)
        )
        
        # Frame buffer for skipped frames
        self.frame_buffer = None
        self.last_processed_frame = None

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from the video source
        Returns:
            Tuple of (success, frame)
        """
        success, frame = self.cap.read()
        
        # Handle None frame
        if not success or frame is None:
            return False, None
        
        # Check if we should process this frame
        if not self.optimizer.start_frame():
            # Skip this frame, return the last processed frame if available
            if self.last_processed_frame is not None:
                return True, self.last_processed_frame.copy()
            return success, frame
        
        # Store the original frame
        self.frame_buffer = frame.copy()
        
        # Resize frame for optimized processing
        frame = cv2.resize(frame, (self.optimized_width, self.optimized_height))
        self.last_processed_frame = frame.copy()
        
        return success, frame

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess the frame for better detection
        Args:
            frame: Input frame
        Returns:
            Preprocessed frame
        """
        # Use the performance optimizer to measure preprocessing time
        @self.optimizer.measure_operation('preprocessing')
        def _preprocess(frame):
            # Resize frame for consistent processing
            frame = cv2.resize(frame, (self.optimized_width, self.optimized_height))
            
            # Convert to RGB (OpenCV uses BGR)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Normalize pixel values
            frame_normalized = frame_rgb / 255.0
            
            return frame_normalized
        
        return _preprocess(frame)

    def draw_results(self, frame: np.ndarray, detections: list) -> np.ndarray:
        """
        Draw detection results on the frame
        Args:
            frame: Input frame
            detections: List of detections
        Returns:
            Frame with drawn results
        """
        # Use the performance optimizer to measure drawing time
        @self.optimizer.measure_operation('drawing')
        def _draw_results(frame, detections):
            # If we're using a resized frame for processing, we need to scale the detections
            scale_x = self.frame_width / self.optimized_width
            scale_y = self.frame_height / self.optimized_height
            
            for detection in detections:
                x1, y1, x2, y2 = detection['bbox']
                label = detection['label']
                confidence = detection['confidence']
                
                # Scale bounding box coordinates
                x1, x2 = x1 * scale_x, x2 * scale_x
                y1, y2 = y1 * scale_y, y2 * scale_y
                
                # Draw bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                # Draw label
                label_text = f"{label}: {confidence:.2f}"
                cv2.putText(frame, label_text, (int(x1), int(y1) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            return frame
        
        # If we're using the optimized frame size, we need to resize back to original size
        if frame.shape[1] != self.frame_width or frame.shape[0] != self.frame_height:
            frame = cv2.resize(frame, (self.frame_width, self.frame_height))
        
        return _draw_results(frame, detections)

    def draw_performance_stats(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw performance statistics on the frame
        Args:
            frame: Input frame
        Returns:
            Frame with performance stats
        """
        stats = self.optimizer.get_performance_stats()
        
        # Draw FPS
        cv2.putText(frame, f"FPS: {stats['fps']:.1f}", (10, self.frame_height - 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw processing quality
        quality = self.optimizer.get_processing_quality()
        cv2.putText(frame, f"Quality: {quality:.2f}", (10, self.frame_height - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame

    def release(self):
        """Release the video capture"""
        self.cap.release()
        cv2.destroyAllWindows() 