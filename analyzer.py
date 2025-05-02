import numpy as np
from typing import List, Dict, Any
from collections import deque
import time

class TrafficAnalyzer:
    def __init__(self, history_size: int = 30):
        """
        Initialize the traffic analyzer
        Args:
            history_size: Number of frames to keep in history
        """
        self.history_size = history_size
        self.vehicle_counts = deque(maxlen=history_size)
        self.timestamps = deque(maxlen=history_size)
        self.start_time = time.time()

    def update(self, detections: List[Dict[str, Any]]):
        """
        Update the analyzer with new detections
        Args:
            detections: List of current frame detections
        """
        current_time = time.time() - self.start_time
        self.timestamps.append(current_time)
        self.vehicle_counts.append(len(detections))

    def get_traffic_density(self) -> float:
        """
        Calculate current traffic density
        Returns:
            Average number of vehicles per frame
        """
        if not self.vehicle_counts:
            return 0.0
        return sum(self.vehicle_counts) / len(self.vehicle_counts)

    def get_traffic_flow_rate(self) -> float:
        """
        Calculate traffic flow rate (vehicles per minute)
        Returns:
            Flow rate in vehicles per minute
        """
        if len(self.timestamps) < 2:
            return 0.0
        
        time_diff = self.timestamps[-1] - self.timestamps[0]
        if time_diff == 0:
            return 0.0
        
        total_vehicles = sum(self.vehicle_counts)
        return (total_vehicles / time_diff) * 60  # Convert to per minute

    def get_traffic_status(self) -> str:
        """
        Get current traffic status based on density and flow rate
        Returns:
            Traffic status description
        """
        density = self.get_traffic_density()
        flow_rate = self.get_traffic_flow_rate()

        if density < 2:
            return "Low Traffic"
        elif density < 5:
            return "Moderate Traffic"
        else:
            return "High Traffic"

    def get_statistics(self) -> Dict[str, float]:
        """
        Get comprehensive traffic statistics
        Returns:
            Dictionary of traffic statistics
        """
        return {
            'traffic_density': self.get_traffic_density(),
            'flow_rate': self.get_traffic_flow_rate(),
            'status': self.get_traffic_status(),
            'total_vehicles': sum(self.vehicle_counts),
            'average_vehicles_per_frame': sum(self.vehicle_counts) / len(self.vehicle_counts) if self.vehicle_counts else 0
        } 