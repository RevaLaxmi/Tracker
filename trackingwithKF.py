import cv2
import numpy as np
import pandas as pd

# Initialize video capture
cap = cv2.VideoCapture('Car-Tracking-Open-CV/videos/top_view_highway.mp4')

# Parameters for filtering contours
min_area = 500
max_area = 3000
min_aspect_ratio = 0.5
max_aspect_ratio = 4.0
min_solidity = 0.7

# Lists for storing tracking data
tracking_data = []
next_vehicle_id = 0
tracked_contours = {}

# Kalman filter setup for each vehicle
vehicle_kalman_filters = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale and apply edge detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
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
                
                # Check if contour matches a previously tracked vehicle
                match_found = False
                for vehicle_id, kalman in vehicle_kalman_filters.items():
                    # Predict the next position of the vehicle
                    prediction = kalman.predict()
                    pred_x, pred_y = int(prediction[0]), int(prediction[1])

                    # Calculate distance to current detected center
                    dist = np.sqrt((pred_x - cx)**2 + (pred_y - cy)**2)
                    if dist < 40:  # Threshold distance for the same vehicle
                        # Update Kalman filter with new position
                        measurement = np.array([[np.float32(cx)], [np.float32(cy)]])
                        kalman.correct(measurement)

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
                
                # Initialize new Kalman filter if no match found
                if not match_found:
                    kalman = cv2.KalmanFilter(4, 2)
                    kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
                    kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
                    kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 5, 0], [0, 0, 0, 5]], np.float32)

                    # Initialize state with first detected position
                    kalman.statePre = np.array([[np.float32(cx)], [np.float32(cy)], [0], [0]], np.float32)
                    kalman.statePost = np.array([[np.float32(cx)], [np.float32(cy)], [0], [0]], np.float32)

                    # Register new vehicle ID and Kalman filter
                    vehicle_id = next_vehicle_id
                    next_vehicle_id += 1
                    vehicle_kalman_filters[vehicle_id] = kalman
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
    cv2.imshow('Kalman Filter Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save the tracking data to a CSV file
df = pd.DataFrame(tracking_data)
df.to_csv('kalman_vehicle_tracking_data.csv', index=False)
