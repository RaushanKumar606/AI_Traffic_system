import time
import threading
from typing import Dict, List, Any, Callable, Tuple
import numpy as np
from functools import wraps

class PerformanceOptimizer:
    def __init__(self, target_fps: int = 30, min_frame_size: Tuple[int, int] = (320, 240)):
        """
        Initialize the performance optimizer
        Args:
            target_fps: Target frames per second
            min_frame_size: Minimum frame size for processing
        """
        self.target_fps = target_fps
        self.frame_time = 1.0 / target_fps
        self.min_frame_size = min_frame_size
        
        # Performance tracking
        self.last_frame_time = time.time()
        self.frame_count = 0
        self.operation_times: Dict[str, list] = {}
        self.processing_quality = 1.0
        
        # Frame skipping
        self.skip_count = 0
        self.max_skip_count = 2
        
        self.frame_times = []
        self.processing_times = {}
        self.start_time = time.time()
        self.performance_stats = {
            'fps': 0,
            'avg_processing_time': 0,
            'detection_time': 0,
            'congestion_time': 0,
            'light_control_time': 0
        }
        self.stats_lock = threading.Lock()
        self.update_interval = 1.0  # seconds
        self.last_stats_update = time.time()
        
        # Adaptive processing parameters
        self.skip_frames = 0
        self.max_skip_frames = 2
        self.min_quality = 0.5
        
        # Performance monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_performance, daemon=True)
        self.monitor_thread.start()

    def start_frame(self) -> bool:
        """
        Determine if the current frame should be processed
        Returns:
            True if frame should be processed, False if it should be skipped
        """
        current_time = time.time()
        elapsed = current_time - self.last_frame_time
        
        # Update frame count and timing
        self.frame_count += 1
        if elapsed >= 1.0:  # Update FPS every second
            self.frame_count = 0
            self.last_frame_time = current_time
        
        # Check if we should skip this frame
        if elapsed < self.frame_time:
            self.skip_count += 1
            if self.skip_count <= self.max_skip_count:
                return False
        
        self.skip_count = 0
        return True

    def measure_operation(self, operation_name: str) -> Callable:
        """
        Decorator to measure operation execution time
        Args:
            operation_name: Name of the operation to measure
        Returns:
            Decorated function
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()
                
                # Record operation time
                if operation_name not in self.operation_times:
                    self.operation_times[operation_name] = []
                self.operation_times[operation_name].append(end_time - start_time)
                
                # Keep only recent measurements
                if len(self.operation_times[operation_name]) > 30:
                    self.operation_times[operation_name].pop(0)
                
                return result
            return wrapper
        return decorator

    def get_performance_stats(self) -> Dict[str, float]:
        """
        Get current performance statistics
        Returns:
            Dictionary of performance metrics
        """
        stats = {
            'fps': self.frame_count,
            'processing_quality': self.processing_quality
        }
        
        # Add average operation times
        for op_name, times in self.operation_times.items():
            if times:
                stats[f'{op_name}_time'] = np.mean(times)
        
        return stats

    def get_processing_quality(self) -> float:
        """
        Get the current processing quality score
        Returns:
            Quality score between 0 and 1
        """
        return self.processing_quality

    def update_processing_quality(self, new_quality: float):
        """
        Update the processing quality score
        Args:
            new_quality: New quality score between 0 and 1
        """
        # Smooth the quality update
        self.processing_quality = 0.8 * self.processing_quality + 0.2 * new_quality

    def get_optimized_frame_size(self, original_size: Tuple[int, int]) -> Tuple[int, int]:
        """
        Calculate the optimized frame size based on performance
        Args:
            original_size: Original frame size (width, height)
        Returns:
            Optimized frame size (width, height)
        """
        width, height = original_size
        
        # Calculate scale factor based on processing quality
        scale_factor = max(0.5, self.processing_quality)
        
        # Calculate new dimensions while maintaining aspect ratio
        new_width = max(self.min_frame_size[0], int(width * scale_factor))
        new_height = max(self.min_frame_size[1], int(height * scale_factor))
        
        return new_width, new_height

    def should_process_detection(self) -> bool:
        """
        Determine if detection should be processed in this frame
        Returns:
            Whether to process detection
        """
        # Process detection every frame if quality is high
        if self.processing_quality > 0.8:
            return True
        
        # Otherwise, process every other frame
        return self.frame_count % 2 == 0

    def should_process_congestion(self) -> bool:
        """
        Determine if congestion analysis should be processed in this frame
        Returns:
            Whether to process congestion analysis
        """
        # Process congestion analysis less frequently
        return self.frame_count % 3 == 0

    def _monitor_performance(self):
        """Monitor and adjust performance parameters"""
        while True:
            time.sleep(self.update_interval)
            
            current_time = time.time()
            elapsed = current_time - self.start_time
            
            with self.stats_lock:
                # Calculate FPS
                if self.frame_count > 0:
                    self.performance_stats['fps'] = self.frame_count / elapsed
                
                # Calculate average processing times
                for op_name, times in self.processing_times.items():
                    if times:
                        avg_time = sum(times) / len(times)
                        self.performance_stats[f'{op_name}_time'] = avg_time
                
                # Adjust processing quality based on performance
                if self.performance_stats['fps'] < self.target_fps * 0.8:
                    # Performance is below target, reduce quality
                    self.processing_quality = max(self.min_quality, self.processing_quality - 0.1)
                elif self.performance_stats['fps'] > self.target_fps * 0.95:
                    # Performance is good, increase quality
                    self.processing_quality = min(1.0, self.processing_quality + 0.05)
                
                # Reset counters for next interval
                self.frame_count = 0
                self.start_time = current_time
                self.last_stats_update = current_time

    def get_performance_stats(self) -> Dict[str, float]:
        """
        Get current performance statistics
        Returns:
            Dictionary of performance statistics
        """
        with self.stats_lock:
            return self.performance_stats.copy() 