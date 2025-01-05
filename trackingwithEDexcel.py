import cv2
import numpy as np
import pandas as pd

# Initialize video capture
cap = cv2.VideoCapture('Car-Tracking-Open-CV/videos/top_view_highway.mp4')

# Parameters for filtering contours
min_area = 500
max_area = 3000
min_aspect_ratio = 0.5 # change
max_aspect_ratio = 4.0
min_solidity = 0.7

# Lists for storing tracking data
tracking_data = []
next_vehicle_id = 0
tracked_contours = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale and apply edge detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    edges = cv2.Canny(gray, 100, 200) # 100 and 200 are the lower and upper threshold for the algo
    # the the hysteresis thresholding algorithm is used to detect edges.
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours
    filtered_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h

            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0

            if min_aspect_ratio < aspect_ratio < max_aspect_ratio and solidity >= min_solidity:
                cx, cy = x + w // 2, y + h // 2  # Center of bounding box
                # Check if the contour matches any previously tracked contours based on position
                match_found = False
                for vehicle_id, (px, py) in tracked_contours.items():
                    dist = np.sqrt((px - cx)**2 + (py - cy)**2)
                    if dist < 40:  # Distance threshold to determine if it's the same vehicle
                        # Update the tracked_contours with new position
                        tracked_contours[vehicle_id] = (cx, cy)
                        # Log tracking data
                        tracking_data.append({
                            'Frame Number': int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
                            'Vehicle ID': vehicle_id,
                            'X-Coordinate': x,
                            'Y-Coordinate': y,
                            'Width': w,
                            'Height': h
                        })
                        match_found = True
                        break
                
                # Assign new ID if no match found
                if not match_found:
                    vehicle_id = next_vehicle_id
                    next_vehicle_id += 1
                    tracked_contours[vehicle_id] = (cx, cy)
                    tracking_data.append({
                        'Frame Number': int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
                        'Vehicle ID': vehicle_id,
                        'X-Coordinate': x,
                        'Y-Coordinate': y,
                        'Width': w,
                        'Height': h
                    })
                
                # Draw bounding box and vehicle ID on frame
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {vehicle_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

    # Display the frame
    cv2.imshow('Contour Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save the tracking data to an Excel file
df = pd.DataFrame(tracking_data)
df.to_csv('vehicle_tracking_data.csv', index=False)
