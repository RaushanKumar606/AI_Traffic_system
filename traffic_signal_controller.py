import time
from enum import Enum
from typing import Dict, List, Tuple

class SignalState(Enum):
    RED = "RED"
    YELLOW = "YELLOW"
    GREEN = "GREEN"

class TrafficSignalController:
    def __init__(self, 
                 min_green_time: float = 10.0,
                 max_green_time: float = 60.0,
                 yellow_time: float = 3.0,
                 red_time: float = 3.0,
                 vehicle_threshold: int = 15):
        """
        Initialize the traffic signal controller
        Args:
            min_green_time: Minimum green light duration in seconds
            max_green_time: Maximum green light duration in seconds
            yellow_time: Yellow light duration in seconds
            red_time: Red light duration in seconds
            vehicle_threshold: Threshold for total vehicles to trigger signal change
        """
        self.min_green_time = min_green_time
        self.max_green_time = max_green_time
        self.yellow_time = yellow_time
        self.red_time = red_time
        self.vehicle_threshold = vehicle_threshold
        
        # Signal states for each direction
        self.signals = {
            'NORTH': SignalState.RED,
            'SOUTH': SignalState.GREEN,
            'EAST': SignalState.RED,
            'WEST': SignalState.RED
        }
        
        # Timing information
        self.last_state_change = time.time()
        self.current_duration = 0.0
        
        # Vehicle counts per direction
        self.vehicle_counts = {
            'NORTH': 0,
            'SOUTH': 0,
            'EAST': 0,
            'WEST': 0
        }
        
        # Emergency vehicle detected
        self.emergency_detected = False
        self.emergency_direction = None

    def update_vehicle_counts(self, counts: Dict[str, int]):
        """
        Update vehicle counts for each direction
        Args:
            counts: Dictionary mapping directions to vehicle counts
        """
        self.vehicle_counts = counts

    def detect_emergency(self, direction: str):
        """
        Handle emergency vehicle detection
        Args:
            direction: Direction of the emergency vehicle
        """
        self.emergency_detected = True
        self.emergency_direction = direction
        # Force green light for emergency vehicle direction
        self.signals[direction] = SignalState.GREEN
        self.last_state_change = time.time()

    def update(self) -> Dict[str, SignalState]:
        """
        Update signal states based on vehicle counts and timing
        Returns:
            Dictionary of current signal states
        """
        current_time = time.time()
        elapsed = current_time - self.last_state_change
        
        # Handle emergency vehicle
        if self.emergency_detected:
            if elapsed >= self.max_green_time:
                self.emergency_detected = False
                self.emergency_direction = None
                self.last_state_change = current_time
            return self.signals
        
        # Calculate total vehicles
        total_vehicles = sum(self.vehicle_counts.values())
        
        # Determine if we need to change signals
        if total_vehicles >= self.vehicle_threshold:
            # Find direction with most vehicles
            max_direction = max(self.vehicle_counts.items(), key=lambda x: x[1])[0]
            
            # Change signals if needed
            if self.signals[max_direction] != SignalState.GREEN:
                # First set current green to yellow
                for direction, state in self.signals.items():
                    if state == SignalState.GREEN:
                        self.signals[direction] = SignalState.YELLOW
                        self.last_state_change = current_time
                        return self.signals
                
                # Then set new direction to green
                if elapsed >= self.yellow_time:
                    self.signals[max_direction] = SignalState.GREEN
                    self.last_state_change = current_time
        
        # Normal signal cycling
        for direction, state in self.signals.items():
            if state == SignalState.GREEN and elapsed >= self.max_green_time:
                self.signals[direction] = SignalState.YELLOW
                self.last_state_change = current_time
            elif state == SignalState.YELLOW and elapsed >= self.yellow_time:
                self.signals[direction] = SignalState.RED
                self.last_state_change = current_time
                # Set next direction to green
                next_direction = self._get_next_direction(direction)
                self.signals[next_direction] = SignalState.GREEN
                self.last_state_change = current_time
        
        return self.signals

    def _get_next_direction(self, current: str) -> str:
        """
        Get the next direction in the signal cycle
        Args:
            current: Current direction
        Returns:
            Next direction
        """
        directions = ['NORTH', 'EAST', 'SOUTH', 'WEST']
        current_index = directions.index(current)
        next_index = (current_index + 1) % len(directions)
        return directions[next_index]

    def get_signal_timing(self) -> Dict[str, float]:
        """
        Get current signal timing information
        Returns:
            Dictionary of timing information
        """
        current_time = time.time()
        elapsed = current_time - self.last_state_change
        
        return {
            'elapsed': elapsed,
            'total_vehicles': sum(self.vehicle_counts.values()),
            'emergency_active': self.emergency_detected
        }

    def draw_signals(self, frame, positions: Dict[str, Tuple[int, int]]):
        """
        Draw traffic signals on the frame
        Args:
            frame: Input frame
            positions: Dictionary mapping directions to (x, y) positions
        Returns:
            Frame with drawn signals
        """
        import cv2
        import numpy as np
        
        # Signal colors
        colors = {
            SignalState.RED: (0, 0, 255),
            SignalState.YELLOW: (0, 255, 255),
            SignalState.GREEN: (0, 255, 0)
        }
        
        # Draw signals
        for direction, pos in positions.items():
            x, y = pos
            state = self.signals[direction]
            color = colors[state]
            
            # Draw signal circle
            cv2.circle(frame, (x, y), 10, color, -1)
            
            # Draw direction label
            cv2.putText(frame, direction, (x - 20, y - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw vehicle count
            count = self.vehicle_counts[direction]
            cv2.putText(frame, f"{count}", (x - 10, y + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame 