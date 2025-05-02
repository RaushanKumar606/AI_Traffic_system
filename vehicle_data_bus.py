from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import time
import threading
from collections import deque

class VehicleType(Enum):
    CAR = "car"
    BUS = "bus"
    BIKE = "bike"
    TRUCK = "truck"
    EMERGENCY = "emergency"

@dataclass
class VehicleData:
    vehicle_id: str
    vehicle_type: VehicleType
    timestamp: float
    position: tuple  # (x, y) coordinates
    speed: float
    direction: float  # angle in degrees
    length: float  # vehicle length in meters
    width: float  # vehicle width in meters
    confidence: float
    lane_id: Optional[str] = None
    is_emergency: bool = False

class VehicleDataBus:
    def __init__(self, max_history: int = 1000):
        """
        Initialize the vehicle data bus
        Args:
            max_history: Maximum number of historical records to keep
        """
        self.vehicle_history: Dict[str, deque] = {}  # vehicle_id -> deque of VehicleData
        self.lane_occupancy: Dict[str, Dict[VehicleType, int]] = {}  # lane_id -> {vehicle_type: count}
        self.max_history = max_history
        self.lock = threading.Lock()
        
        # Statistics
        self.total_vehicles = 0
        self.vehicle_counts = {vt: 0 for vt in VehicleType}
        self.avg_speeds = {vt: 0.0 for vt in VehicleType}
        self.speed_counts = {vt: 0 for vt in VehicleType}

    def add_vehicle_data(self, data: VehicleData):
        """
        Add new vehicle data to the bus
        Args:
            data: VehicleData object containing vehicle information
        """
        with self.lock:
            # Update vehicle history
            if data.vehicle_id not in self.vehicle_history:
                self.vehicle_history[data.vehicle_id] = deque(maxlen=self.max_history)
            self.vehicle_history[data.vehicle_id].append(data)
            
            # Update lane occupancy
            if data.lane_id:
                if data.lane_id not in self.lane_occupancy:
                    self.lane_occupancy[data.lane_id] = {vt: 0 for vt in VehicleType}
                self.lane_occupancy[data.lane_id][data.vehicle_type] += 1
            
            # Update statistics
            self.total_vehicles += 1
            self.vehicle_counts[data.vehicle_type] += 1
            
            # Update average speed
            self.avg_speeds[data.vehicle_type] = (
                (self.avg_speeds[data.vehicle_type] * self.speed_counts[data.vehicle_type] + data.speed) /
                (self.speed_counts[data.vehicle_type] + 1)
            )
            self.speed_counts[data.vehicle_type] += 1

    def get_vehicle_history(self, vehicle_id: str) -> List[VehicleData]:
        """
        Get historical data for a specific vehicle
        Args:
            vehicle_id: ID of the vehicle
        Returns:
            List of historical VehicleData objects
        """
        with self.lock:
            return list(self.vehicle_history.get(vehicle_id, []))

    def get_lane_occupancy(self, lane_id: str) -> Dict[VehicleType, int]:
        """
        Get vehicle counts by type for a specific lane
        Args:
            lane_id: ID of the lane
        Returns:
            Dictionary mapping vehicle types to counts
        """
        with self.lock:
            return self.lane_occupancy.get(lane_id, {vt: 0 for vt in VehicleType})

    def get_vehicle_type_stats(self) -> Dict[VehicleType, Dict[str, float]]:
        """
        Get statistics for each vehicle type
        Returns:
            Dictionary mapping vehicle types to their statistics
        """
        with self.lock:
            stats = {}
            for vt in VehicleType:
                stats[vt] = {
                    'count': self.vehicle_counts[vt],
                    'avg_speed': self.avg_speeds[vt],
                    'percentage': (self.vehicle_counts[vt] / self.total_vehicles * 100 
                                 if self.total_vehicles > 0 else 0)
                }
            return stats

    def get_emergency_vehicles(self) -> List[VehicleData]:
        """
        Get list of emergency vehicles in the system
        Returns:
            List of VehicleData objects for emergency vehicles
        """
        with self.lock:
            emergency_vehicles = []
            for vehicle_history in self.vehicle_history.values():
                if vehicle_history and vehicle_history[-1].is_emergency:
                    emergency_vehicles.append(vehicle_history[-1])
            return emergency_vehicles

    def clear_old_data(self, max_age: float = 60.0):
        """
        Clear data older than max_age seconds
        Args:
            max_age: Maximum age of data in seconds
        """
        current_time = time.time()
        with self.lock:
            for vehicle_id, history in list(self.vehicle_history.items()):
                # Remove old entries
                while history and (current_time - history[0].timestamp) > max_age:
                    history.popleft()
                
                # Remove empty histories
                if not history:
                    del self.vehicle_history[vehicle_id]

    def get_traffic_density(self, lane_id: Optional[str] = None) -> float:
        """
        Calculate traffic density for a specific lane or all lanes
        Args:
            lane_id: Optional lane ID to calculate density for
        Returns:
            Traffic density (vehicles per kilometer)
        """
        with self.lock:
            if lane_id:
                # Calculate density for specific lane
                lane_data = self.lane_occupancy.get(lane_id, {})
                total_vehicles = sum(lane_data.values())
                # Assuming standard lane length of 1 kilometer
                return total_vehicles
            else:
                # Calculate overall density
                total_vehicles = sum(self.vehicle_counts.values())
                # Assuming total road length of 1 kilometer
                return total_vehicles

    def get_vehicle_type_distribution(self) -> Dict[VehicleType, float]:
        """
        Get the distribution of vehicle types in the system
        Returns:
            Dictionary mapping vehicle types to their percentage
        """
        with self.lock:
            if self.total_vehicles == 0:
                return {vt: 0.0 for vt in VehicleType}
            
            return {
                vt: (self.vehicle_counts[vt] / self.total_vehicles * 100)
                for vt in VehicleType
            } 