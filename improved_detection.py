import cv2
import numpy as np
import pandas as pd

# Initialize video capture
cap = cv2.VideoCapture('Car-Tracking-Open-CV/videos/top_view_highway.mp4')

# Background subtractor
back_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

# Initial ROI parameters
initial_roi_y = 0.5  # Start from halfway down the frame
adaptive_roi_margin = 50  # Margin to expand the ROI based on detections

# Contour filtering parameters
min_area = 500
max_area = 3000
min_aspect_ratio = 0.5
max_aspect_ratio = 4.0
min_solidity = 0.7

'''
if we want to apply this to everything we need to lose the filtering parameters
and just use the adaptive ROI
with OBR globally not based in a certain contour area 
'''

# Tracking data
tracking_data = []
next_vehicle_id = 0
tracked_contours = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Define initial ROI dimensions
    height, width = frame.shape[:2]
    roi_y = int(height * initial_roi_y)  # Start ROI at initial percentage
    roi_height = height - roi_y          # ROI covers the lower portion of the frame

    # Apply Gaussian blur and background subtraction
    frame_blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    fg_mask = back_subtractor.apply(frame_blurred)

    # Extract ROI for contour detection
    roi = fg_mask[roi_y:height, :]

    # Find contours within ROI
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Update tracking info
    current_frame_contours = {}

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h

            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0

            if min_aspect_ratio < aspect_ratio < max_aspect_ratio and solidity >= min_solidity:
                global_y = y + roi_y  # Adjust y-coordinates relative to original frame
                cx, cy = x + w // 2, global_y + h // 2

                # Track existing or assign new ID
                match_found = False
                for vehicle_id, (px, py) in tracked_contours.items():
                    dist = np.sqrt((px - cx)**2 + (py - cy)**2)
                    if dist < 50:  # Distance threshold for matching
                        tracked_contours[vehicle_id] = (cx, cy)
                        current_frame_contours[vehicle_id] = (x, global_y, w, h)
                        match_found = True
                        break
                
                if not match_found:
                    vehicle_id = next_vehicle_id
                    next_vehicle_id += 1
                    tracked_contours[vehicle_id] = (cx, cy)
                    current_frame_contours[vehicle_id] = (x, global_y, w, h)

                # Draw bounding box and vehicle ID
                cv2.rectangle(frame, (x, global_y), (x + w, global_y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {vehicle_id}", (x, global_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Expand ROI if vehicles are detected near its boundary
    if current_frame_contours:
        min_y = min([bbox[1] for bbox in current_frame_contours.values()])
        if min_y < roi_y:
            roi_y = max(0, min_y - adaptive_roi_margin)

    # Display the frame
    cv2.imshow('Adaptive ROI Car Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
