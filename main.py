import cv2
import numpy as np
from video_processor import VideoProcessor
from vehicle_data_bus import VehicleDataBus, VehicleType, VehicleData
from traffic_signal_controller import TrafficSignalController
from detector import VehicleDetector
import time
import uuid
import os
import datetime
from collections import deque
import csv

class TrafficAnalyzer:
    def __init__(self, history_size=300):  # 10 seconds at 30 FPS
        self.history_size = history_size
        self.vehicle_counts = deque(maxlen=history_size)
        self.timestamps = deque(maxlen=history_size)
        self.direction_counts = {
            'NORTH': deque(maxlen=history_size),
            'SOUTH': deque(maxlen=history_size),
            'EAST': deque(maxlen=history_size),
            'WEST': deque(maxlen=history_size)
        }
        self.start_time = time.time()
        
    def update(self, direction_counts):
        current_time = time.time() - self.start_time
        self.timestamps.append(current_time)
        total_vehicles = sum(direction_counts.values())
        self.vehicle_counts.append(total_vehicles)
        
        for direction, count in direction_counts.items():
            self.direction_counts[direction].append(count)
    
    def get_traffic_density(self):
        if not self.vehicle_counts:
            return 0.0
        return sum(self.vehicle_counts) / len(self.vehicle_counts)
    
    def get_direction_density(self, direction):
        if not self.direction_counts[direction]:
            return 0.0
        return sum(self.direction_counts[direction]) / len(self.direction_counts[direction])
    
    def get_signal_timing(self, direction):
        density = self.get_direction_density(direction)
        # Base timing: 30 seconds
        # Maximum timing: 90 seconds
        # Scale based on density (0-10 vehicles)
        base_time = 30
        max_time = 90
        density_factor = min(density / 10.0, 1.0)
        return int(base_time + (max_time - base_time) * density_factor)
    
    def get_traffic_flow_rate(self):
        if len(self.timestamps) < 2:
            return 0.0
        time_diff = self.timestamps[-1] - self.timestamps[0]
        if time_diff == 0:
            return 0.0
        total_vehicles = sum(self.vehicle_counts)
        return (total_vehicles / time_diff) * 60  # vehicles per minute

def save_vehicle_data_to_csv(vehicle_data, direction_counts, signal_times, timestamp):
    """
    Save vehicle data to CSV file
    Args:
        vehicle_data: List of VehicleData objects
        direction_counts: Dictionary of vehicle counts by direction
        signal_times: Dictionary of signal times by direction
        timestamp: Current timestamp
    """
    filename = "vehicle_data.csv"
    file_exists = os.path.exists(filename)
    
    with open(filename, 'a', newline='') as csvfile:
        fieldnames = [
            'timestamp',
            'total_vehicles',
            'north_vehicles', 'north_signal_time',
            'south_vehicles', 'south_signal_time',
            'east_vehicles', 'east_signal_time',
            'west_vehicles', 'west_signal_time',
            'car_count', 'bus_count', 'bike_count', 'truck_count', 'emergency_count',
            'traffic_density',
            'flow_rate'
        ]
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        # Count vehicles by type
        type_counts = {vt: 0 for vt in VehicleType}
        for data in vehicle_data:
            type_counts[data.vehicle_type] += 1
        
        # Write data row
        writer.writerow({
            'timestamp': timestamp,
            'total_vehicles': sum(direction_counts.values()),
            'north_vehicles': direction_counts['NORTH'],
            'north_signal_time': signal_times['NORTH'],
            'south_vehicles': direction_counts['SOUTH'],
            'south_signal_time': signal_times['SOUTH'],
            'east_vehicles': direction_counts['EAST'],
            'east_signal_time': signal_times['EAST'],
            'west_vehicles': direction_counts['WEST'],
            'west_signal_time': signal_times['WEST'],
            'car_count': type_counts[VehicleType.CAR],
            'bus_count': type_counts[VehicleType.BUS],
            'bike_count': type_counts[VehicleType.BIKE],
            'truck_count': type_counts[VehicleType.TRUCK],
            'emergency_count': type_counts[VehicleType.EMERGENCY],
            'traffic_density': sum(direction_counts.values()) / 4,  # Average per direction
            'flow_rate': sum(direction_counts.values()) / 60  # Vehicles per second
        })

def main():
    # Use the traffic.mp4 file that's already in the workspace
    video_source = "traffic.mp4"
    
    # Check if video file exists
    if not os.path.exists(video_source):
        print(f"Error: Video file '{video_source}' not found.")
        print("Please provide a valid traffic video file path.")
        return
    
    # Initialize video processor with the traffic video
    processor = VideoProcessor(source=video_source, target_fps=30)
    
    # Initialize vehicle detector
    detector = VehicleDetector()
    
    # Initialize vehicle data bus
    data_bus = VehicleDataBus(max_history=1000)
    
    # Initialize traffic signal controller
    signal_controller = TrafficSignalController(vehicle_threshold=15)
    
    # Initialize traffic analyzer
    traffic_analyzer = TrafficAnalyzer()
    
    # Define signal positions (adjust these based on your video)
    frame_height, frame_width = processor.frame_height, processor.frame_width
    signal_positions = {
        'NORTH': (frame_width // 2, 30),
        'SOUTH': (frame_width // 2, frame_height - 30),
        'EAST': (frame_width - 30, frame_height // 2),
        'WEST': (30, frame_height // 2)
    }
    
    # Time tracking
    start_time = time.time()
    frame_count = 0
    last_save_time = time.time()
    save_interval = 1.0  # Save data every second
    
    print(f"Processing traffic video: {video_source}")
    print(f"Video resolution: {frame_width}x{frame_height}")
    print("Press 'q' to quit")
    
    try:
        while True:
            # Read frame
            success, frame = processor.read_frame()
            if not success:
                print("End of video reached. Restarting...")
                # Reset video to beginning
                processor.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            # Update frame count and elapsed time
            frame_count += 1
            elapsed_time = time.time() - start_time
            current_time = datetime.datetime.now()
            
            # Preprocess frame
            processed_frame = processor.preprocess_frame(frame)
            
            # Detect vehicles in the frame
            detections = detector.detect(processed_frame, confidence_threshold=0.5)
            
            # Process detections and update data bus
            vehicle_data_list = []
            for detection in detections:
                vehicle_data = create_vehicle_data(detection)
                data_bus.add_vehicle_data(vehicle_data)
                vehicle_data_list.append(vehicle_data)
            
            # Update vehicle counts for traffic signals
            direction_counts = {
                'NORTH': count_vehicles_in_direction(detections, 'NORTH', frame_height, frame_width),
                'SOUTH': count_vehicles_in_direction(detections, 'SOUTH', frame_height, frame_width),
                'EAST': count_vehicles_in_direction(detections, 'EAST', frame_height, frame_width),
                'WEST': count_vehicles_in_direction(detections, 'WEST', frame_height, frame_width)
            }
            
            # Update traffic analyzer
            traffic_analyzer.update(direction_counts)
            
            # Get signal times for each direction
            signal_times = {
                'NORTH': traffic_analyzer.get_signal_timing('NORTH'),
                'SOUTH': traffic_analyzer.get_signal_timing('SOUTH'),
                'EAST': traffic_analyzer.get_signal_timing('EAST'),
                'WEST': traffic_analyzer.get_signal_timing('WEST')
            }
            
            # Update traffic signals with dynamic timing
            signal_controller.update_vehicle_counts(direction_counts)
            signal_controller.update()
            
            # Save data to CSV periodically
            current_time = time.time()
            if current_time - last_save_time >= save_interval:
                save_vehicle_data_to_csv(vehicle_data_list, direction_counts, signal_times, 
                                       datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                last_save_time = current_time
            
            # Draw results
            frame = processor.draw_results(frame, detections)
            
            # Draw traffic signals
            frame = signal_controller.draw_signals(frame, signal_positions)
            
            # Draw vehicle statistics
            frame = draw_vehicle_stats(frame, data_bus)
            
            # Draw time and NEWS information with dynamic timing
            frame = draw_time_and_news(frame, elapsed_time, direction_counts, frame_count, traffic_analyzer)
            
            # Draw performance stats
            frame = processor.draw_performance_stats(frame)
            
            # Display frame
            cv2.imshow('Traffic Monitoring', frame)
            
            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Clear old data periodically
            data_bus.clear_old_data(max_age=60.0)
    
    finally:
        processor.release()
        cv2.destroyAllWindows()

def count_vehicles_in_direction(detections, direction, frame_height, frame_width):
    """
    Count vehicles in a specific direction
    """
    count = 0
    if direction == 'NORTH':
        region = (0, 0, frame_width, frame_height // 3)
    elif direction == 'SOUTH':
        region = (0, 2 * frame_height // 3, frame_width, frame_height)
    elif direction == 'EAST':
        region = (2 * frame_width // 3, 0, frame_width, frame_height)
    else:  # WEST
        region = (0, 0, frame_width // 3, frame_height)
    
    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        if (region[0] <= center_x <= region[2] and 
            region[1] <= center_y <= region[3]):
            count += 1
    
    return count

def create_vehicle_data(detection):
    """
    Create VehicleData object from detection
    """
    x1, y1, x2, y2 = detection['bbox']
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    # Calculate approximate speed (in a real system, this would use tracking)
    speed = np.random.uniform(0, 60)  # km/h
    
    # Calculate approximate direction (in a real system, this would use tracking)
    direction = np.random.uniform(0, 360)  # degrees
    
    # Calculate approximate dimensions (in meters)
    width_pixels = x2 - x1
    height_pixels = y2 - y1
    # Assuming 1 meter = 50 pixels (this would need calibration in a real system)
    width_meters = width_pixels / 50
    height_meters = height_pixels / 50
    
    # Map detection label to VehicleType
    vehicle_type_map = {
        'car': VehicleType.CAR,
        'bus': VehicleType.BUS,
        'motorcycle': VehicleType.BIKE,
        'truck': VehicleType.TRUCK,
        'ambulance': VehicleType.EMERGENCY
    }
    
    vehicle_type = vehicle_type_map.get(detection['label'], VehicleType.CAR)
    
    return VehicleData(
        vehicle_id=str(uuid.uuid4()),
        vehicle_type=vehicle_type,
        timestamp=time.time(),
        position=(center_x, center_y),
        speed=speed,
        direction=direction,
        length=height_meters,
        width=width_meters,
        confidence=detection['confidence'],
        lane_id=f"lane_{np.random.randint(1, 4)}",  # Simulate 3 lanes
        is_emergency=vehicle_type == VehicleType.EMERGENCY
    )

def draw_vehicle_stats(frame, data_bus):
    """
    Draw vehicle statistics on the frame
    """
    # Get vehicle type statistics
    stats = data_bus.get_vehicle_type_stats()
    
    # Draw vehicle counts
    y_offset = 30
    for vt, stat in stats.items():
        if stat['count'] > 0:  # Only show vehicle types that are present
            cv2.putText(frame, f"{vt.value}: {stat['count']} ({stat['percentage']:.1f}%)",
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 25
    
    # Draw traffic density
    density = data_bus.get_traffic_density()
    cv2.putText(frame, f"Traffic Density: {density:.1f} vehicles/km",
               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return frame

def draw_time_and_news(frame, elapsed_time, direction_counts, frame_count, traffic_analyzer):
    """
    Draw time and NEWS information on the frame with dynamic timing
    """
    # Draw current time
    current_time = datetime.datetime.now().strftime("%H:%M:%S")
    cv2.putText(frame, f"Time: {current_time}", 
               (frame.shape[1] - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Draw elapsed time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    cv2.putText(frame, f"Elapsed: {minutes:02d}:{seconds:02d}", 
               (frame.shape[1] - 200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Draw frame count
    cv2.putText(frame, f"Frames: {frame_count}", 
               (frame.shape[1] - 200, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Draw NEWS information with dynamic timing
    y_offset = 120
    for direction, count in direction_counts.items():
        # Get recommended signal timing for this direction
        signal_time = traffic_analyzer.get_signal_timing(direction)
        
        # Draw direction count and timing
        cv2.putText(frame, f"{direction}: {count} vehicles ({signal_time}s)", 
                   (frame.shape[1] - 200, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += 30
    
    # Draw total vehicles and traffic flow rate
    total_vehicles = sum(direction_counts.values())
    flow_rate = traffic_analyzer.get_traffic_flow_rate()
    cv2.putText(frame, f"Total: {total_vehicles} vehicles", 
               (frame.shape[1] - 200, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    y_offset += 30
    cv2.putText(frame, f"Flow Rate: {flow_rate:.1f} vehicles/min", 
               (frame.shape[1] - 200, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return frame

if __name__ == "__main__":
    main() 