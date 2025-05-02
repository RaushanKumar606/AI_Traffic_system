import cv2
import numpy as np
from typing import Dict, List, Any
import time
from congestion_predictor import CongestionPredictor

class TrafficLightController:
    def __init__(self):
        self.LIGHT_STATES = {
            'RED': (0, 0, 255),
            'YELLOW': (0, 255, 255),
            'GREEN': (0, 255, 0)
        }
        
        # Initialize roads (North, South, East, West)
        self.roads = {
            'NORTH': {'state': 'RED', 'timer': 0, 'vehicles': 0},
            'SOUTH': {'state': 'RED', 'timer': 0, 'vehicles': 0},
            'EAST': {'state': 'GREEN', 'timer': 0, 'vehicles': 0},
            'WEST': {'state': 'RED', 'timer': 0, 'vehicles': 0}
        }
        
        # Initialize congestion predictor
        self.congestion_predictor = CongestionPredictor()
        
        # Timing parameters (in seconds)
        self.MIN_GREEN_TIME = 10
        self.MAX_GREEN_TIME = 45
        self.YELLOW_TIME = 3
        self.last_update = time.time()
        
        # Current active road
        self.active_road = 'EAST'
        
        # Emergency vehicle tracking
        self.emergency_active = False
        self.emergency_road = None
        self.emergency_start_time = 0
        self.EMERGENCY_DURATION = 15  # seconds

    def calculate_green_time(self, road: str) -> int:
        """Calculate green time based on congestion prediction"""
        timing = self.congestion_predictor.get_optimal_signal_timing(road)
        return timing['green_time']

    def update(self, detections: List[Dict[str, Any]], road_regions: Dict[str, List[int]]):
        """
        Update traffic light states based on detections
        Args:
            detections: List of vehicle detections
            road_regions: Dictionary mapping road names to their pixel regions
        """
        current_time = time.time()
        elapsed_time = current_time - self.last_update
        self.last_update = current_time

        # Count vehicles in each road region
        road_counts = {road: 0 for road in self.roads}
        has_ambulance = {road: False for road in self.roads}

        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Check which road region this detection belongs to
            for road, region in road_regions.items():
                x_min, y_min, x_max, y_max = region
                if x_min <= center_x <= x_max and y_min <= center_y <= y_max:
                    road_counts[road] += 1
                    # Check if it's an ambulance
                    if detection['label'] == 'ambulance':
                        has_ambulance[road] = True

        # Update vehicle counts
        for road in self.roads:
            self.roads[road]['vehicles'] = road_counts[road]
        
        # Update congestion predictor
        self.congestion_predictor.update(road_counts)

        # Handle emergency vehicle priority
        for road, has_emergency in has_ambulance.items():
            if has_emergency and not self.emergency_active:
                self._activate_emergency(road)
                return
        
        # Check if emergency mode should end
        if self.emergency_active:
            if current_time - self.emergency_start_time >= self.EMERGENCY_DURATION:
                self._deactivate_emergency()
            else:
                return  # Continue emergency mode

        # Normal traffic light control with congestion-based timing
        active_road = self.active_road
        active_state = self.roads[active_road]['state']
        active_timer = self.roads[active_road]['timer']

        if active_state == 'GREEN':
            green_time = self.calculate_green_time(active_road)
            if active_timer >= green_time:
                self._switch_to_yellow(active_road)
            else:
                self.roads[active_road]['timer'] += elapsed_time

        elif active_state == 'YELLOW':
            if active_timer >= self.YELLOW_TIME:
                self._switch_to_next_road()
            else:
                self.roads[active_road]['timer'] += elapsed_time

    def _activate_emergency(self, emergency_road: str):
        """Activate emergency mode for ambulance"""
        self.emergency_active = True
        self.emergency_road = emergency_road
        self.emergency_start_time = time.time()
        
        # Set all roads to red except emergency road
        for road in self.roads:
            if road == emergency_road:
                self.roads[road]['state'] = 'GREEN'
                self.roads[road]['timer'] = 0
            else:
                self.roads[road]['state'] = 'RED'
                self.roads[road]['timer'] = 0
        self.active_road = emergency_road

    def _deactivate_emergency(self):
        """Deactivate emergency mode"""
        self.emergency_active = False
        self.emergency_road = None
        
        # Reset to normal operation
        self._switch_to_next_road()

    def _switch_to_yellow(self, road: str):
        """Switch specified road to yellow"""
        self.roads[road]['state'] = 'YELLOW'
        self.roads[road]['timer'] = 0

    def _switch_to_next_road(self):
        """Switch to next road in sequence based on congestion priority"""
        # Get road priorities from congestion predictor
        priorities = self.congestion_predictor.get_road_priorities()
        
        # Find road with highest priority
        next_road = max(priorities, key=priorities.get)
        
        # Set all roads to red
        for road in self.roads:
            self.roads[road]['state'] = 'RED'
            self.roads[road]['timer'] = 0

        # Set next road to green
        self.roads[next_road]['state'] = 'GREEN'
        self.active_road = next_road

    def draw_traffic_lights(self, frame: np.ndarray, positions: Dict[str, tuple]) -> np.ndarray:
        """
        Draw traffic lights on the frame
        Args:
            frame: Input frame
            positions: Dictionary mapping road names to light positions
        Returns:
            Frame with traffic lights drawn
        """
        for road, pos in positions.items():
            state = self.roads[road]['state']
            color = self.LIGHT_STATES[state]
            vehicles = self.roads[road]['vehicles']
            
            # Get congestion prediction
            prediction = self.congestion_predictor.predict_congestion(road)
            
            # Draw traffic light circle
            cv2.circle(frame, pos, 20, color, -1)
            
            # Draw vehicle count and congestion info
            text_pos = (pos[0] - 20, pos[1] + 40)
            cv2.putText(frame, f"{road}: {vehicles}", text_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Draw congestion level
            congestion_pos = (pos[0] - 20, pos[1] + 60)
            cv2.putText(frame, f"Cong: {prediction['current_level']}", congestion_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Draw timer if green
            if state == 'GREEN':
                timer_pos = (pos[0] - 20, pos[1] + 80)
                remaining = self.calculate_green_time(road) - self.roads[road]['timer']
                cv2.putText(frame, f"Time: {int(remaining)}s", timer_pos,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Draw emergency indicator if active
            if self.emergency_active and road == self.emergency_road:
                emergency_pos = (pos[0] - 20, pos[1] + 100)
                cv2.putText(frame, "EMERGENCY", emergency_pos,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return frame 