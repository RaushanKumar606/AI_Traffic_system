import cv2
import numpy as np

def main():
    # Open the video file
    cap = cv2.VideoCapture("traffic.mp4")
    
    if not cap.isOpened():
        print("Error: Could not open video file")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"Video properties: {width}x{height} @ {fps}fps")
    
    # Read and display the first frame
    ret, frame = cap.read()
    if ret:
        print("Successfully read the first frame")
        cv2.imshow("First Frame", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Failed to read the first frame")
    
    # Release resources
    cap.release()

if __name__ == "__main__":
    main() 