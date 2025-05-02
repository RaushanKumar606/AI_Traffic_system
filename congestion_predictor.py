import numpy as np
from typing import Dict, List, Any
import time
from collections import deque

class CongestionPredictor:
    def __init__(self, history_size: int = 60):
        """
        Initialize the congestion predictor
        Args:
            history_size: Number of frames to keep in history
        """
        self.history_size = history_size
        self.vehicle_history = {road: deque(maxlen=history_size) for road in ['NORTH', 'SOUTH', 'EAST', 'WEST']}
        self.congestion_thresholds = {
            'LOW': 3,      # Less than 3 vehicles
            'MODERATE': 7, # 3-7 vehicles
            'HIGH': 10     # More than 10 vehicles
        }
        self.congestion_levels = {
            'LOW': 0,
            'MODERATE': 1,
            'HIGH': 2,
            'SEVERE': 3
        }
        self.last_prediction_time = time.time()
        self.prediction_interval = 5  # seconds

    def update(self, road_counts: Dict[str, int]):
        """
        Update the predictor with new vehicle counts
        Args:
            road_counts: Dictionary mapping road names to vehicle counts
        """
        for road, count in road_counts.items():
            self.vehicle_history[road].append(count)

    def get_congestion_level(self, vehicle_count: int) -> str:
        """
        Determine congestion level based on vehicle count
        Args:
            vehicle_count: Number of vehicles
        Returns:
            Congestion level string
        """
        if vehicle_count < self.congestion_thresholds['LOW']:
            return 'LOW'
        elif vehicle_count < self.congestion_thresholds['MODERATE']:
            return 'MODERATE'
        elif vehicle_count < self.congestion_thresholds['HIGH']:
            return 'HIGH'
        else:
            return 'SEVERE'

    def predict_congestion(self, road: str) -> Dict[str, Any]:
        """
        Predict congestion for a specific road
        Args:
            road: Road name
        Returns:
            Dictionary with congestion prediction
        """
        if len(self.vehicle_history[road]) < 5:
            return {
                'current_level': 'UNKNOWN',
                'predicted_level': 'UNKNOWN',
                'trend': 'STABLE',
                'confidence': 0.0
            }

        # Calculate current congestion level
        current_count = self.vehicle_history[road][-1]
        current_level = self.get_congestion_level(current_count)
        
        # Calculate trend
        recent_counts = list(self.vehicle_history[road])[-5:]
        trend = 'STABLE'
        if len(recent_counts) >= 2:
            if recent_counts[-1] > recent_counts[-2] * 1.2:  # 20% increase
                trend = 'INCREASING'
            elif recent_counts[-1] < recent_counts[-2] * 0.8:  # 20% decrease
                trend = 'DECREASING'
        
        # Predict future congestion
        predicted_level = current_level
        confidence = 0.7  # Base confidence
        
        # Adjust prediction based on trend
        if trend == 'INCREASING':
            if current_level == 'LOW':
                predicted_level = 'MODERATE'
            elif current_level == 'MODERATE':
                predicted_level = 'HIGH'
            elif current_level == 'HIGH':
                predicted_level = 'SEVERE'
            confidence += 0.1
        elif trend == 'DECREASING':
            if current_level == 'SEVERE':
                predicted_level = 'HIGH'
            elif current_level == 'HIGH':
                predicted_level = 'MODERATE'
            elif current_level == 'MODERATE':
                predicted_level = 'LOW'
            confidence += 0.1
        
        return {
            'current_level': current_level,
            'predicted_level': predicted_level,
            'trend': trend,
            'confidence': min(confidence, 0.95)
        }

    def get_road_priorities(self) -> Dict[str, float]:
        """
        Calculate priority scores for each road
        Returns:
            Dictionary mapping road names to priority scores
        """
        priorities = {}
        for road in self.vehicle_history:
            prediction = self.predict_congestion(road)
            
            # Base priority on congestion level
            level_score = self.congestion_levels[prediction['current_level']]
            
            # Adjust based on trend
            trend_multiplier = 1.0
            if prediction['trend'] == 'INCREASING':
                trend_multiplier = 1.2
            elif prediction['trend'] == 'DECREASING':
                trend_multiplier = 0.8
            
            # Calculate final priority score
            priorities[road] = level_score * trend_multiplier * prediction['confidence']
        
        return priorities

    def get_optimal_signal_timing(self, road: str) -> Dict[str, int]:
        """
        Calculate optimal signal timing based on congestion prediction
        Args:
            road: Road name
        Returns:
            Dictionary with signal timing parameters
        """
        prediction = self.predict_congestion(road)
        current_count = self.vehicle_history[road][-1] if self.vehicle_history[road] else 0
        
        # Base timing parameters
        base_green_time = 15
        base_yellow_time = 3
        
        # Adjust based on congestion level
        if prediction['current_level'] == 'LOW':
            green_time = base_green_time
        elif prediction['current_level'] == 'MODERATE':
            green_time = base_green_time + 5
        elif prediction['current_level'] == 'HIGH':
            green_time = base_green_time + 10
        else:  # SEVERE
            green_time = base_green_time + 15
        
        # Further adjust based on vehicle count
        green_time += min(current_count, 10)  # Add up to 10 seconds based on vehicle count
        
        # Cap maximum green time
        max_green_time = 45
        green_time = min(green_time, max_green_time)
        
        return {
            'green_time': green_time,
            'yellow_time': base_yellow_time
        } 