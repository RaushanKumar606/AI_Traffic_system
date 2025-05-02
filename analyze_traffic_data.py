import pandas as pd
import matplotlib.pyplot as plt
import datetime

def analyze_traffic_data(csv_file="vehicle_data.csv"):
    """
    Analyze traffic data from CSV file
    """
    try:
        # Read the CSV file with explicit encoding
        print("Reading traffic data...")
        df = pd.read_csv(csv_file, encoding='utf-8')
        print(f"Successfully read {len(df)} records")
        
        # Basic statistics
        print("\n=== Traffic Statistics ===")
        print(f"Total Records: {len(df)}")
        print(f"Average Total Vehicles: {df['total_vehicles'].mean():.1f}")
        print(f"Maximum Total Vehicles: {df['total_vehicles'].max()}")
        
        # Vehicle counts by direction
        print("\n=== Vehicle Counts by Direction ===")
        for direction in ['north', 'south', 'east', 'west']:
            vehicles = df[f'{direction}_vehicles'].mean()
            signal_time = df[f'{direction}_signal_time'].mean()
            print(f"{direction.upper()}:")
            print(f"  Average Vehicles: {vehicles:.1f}")
            print(f"  Average Signal Time: {signal_time:.1f} seconds")
        
        # Vehicle type distribution
        print("\n=== Vehicle Type Distribution ===")
        for vtype in ['car', 'bus', 'bike', 'truck', 'emergency']:
            count = df[f'{vtype}_count'].mean()
            total = df[f'{vtype}_count'].sum()
            print(f"{vtype.title()}:")
            print(f"  Average: {count:.1f}")
            print(f"  Total: {total}")
        
        # Create plots
        plt.figure(figsize=(12, 6))
        
        # Plot 1: Direction-wise vehicles
        plt.subplot(1, 2, 1)
        directions = ['north', 'south', 'east', 'west']
        vehicles = [df[f'{d}_vehicles'].mean() for d in directions]
        plt.bar(directions, vehicles)
        plt.title('Average Vehicles by Direction')
        plt.ylabel('Number of Vehicles')
        
        # Plot 2: Vehicle types
        plt.subplot(1, 2, 2)
        types = ['car', 'bus', 'bike', 'truck', 'emergency']
        counts = [df[f'{t}_count'].mean() for t in types]
        plt.bar(types, counts)
        plt.title('Average Vehicles by Type')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('traffic_analysis.png')
        print("\nPlots saved as 'traffic_analysis.png'")
        
        # Traffic recommendations
        print("\n=== Traffic Recommendations ===")
        busiest_direction = max(directions, key=lambda d: df[f'{d}_vehicles'].mean())
        print(f"Busiest Direction: {busiest_direction.upper()}")
        print(f"Recommended Actions:")
        print(f"1. Increase signal time for {busiest_direction.upper()} direction")
        print(f"2. Monitor {busiest_direction.upper()} traffic flow")
        
        if df['emergency_count'].sum() > 0:
            print("3. Emergency vehicles detected - ensure priority signaling")
        
        # Time-based analysis
        print("\n=== Time-Based Analysis ===")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        hourly_traffic = df.groupby('hour')['total_vehicles'].mean()
        peak_hour = hourly_traffic.idxmax()
        print(f"Peak Hour: {peak_hour:02d}:00")
        print(f"Average Vehicles at Peak: {hourly_traffic[peak_hour]:.1f}")
        
    except FileNotFoundError:
        print(f"Error: {csv_file} not found.")
        print("Please make sure the traffic monitoring system has generated data.")
    except Exception as e:
        print(f"Error analyzing data: {str(e)}")
        print("Data sample:")
        try:
            with open(csv_file, 'r') as f:
                print(f.read()[:500])
        except:
            pass

if __name__ == "__main__":
    analyze_traffic_data() 