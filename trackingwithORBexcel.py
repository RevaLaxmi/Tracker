import cv2
import numpy as np
import csv

# Initialize video capture
cap = cv2.VideoCapture("cpy\car_in_dirt.mp4")

# ORB detector and Brute-Force Matcher setup
orb = cv2.ORB_create(nfeatures=2000)  # Increase the feature count if necessary
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Parameters for contour filtering
min_area = 500
max_area = 3000
min_aspect_ratio = 0.5
max_aspect_ratio = 4.0
min_solidity = 0.7

# Store previous frame's data
previous_keypoints = []
previous_descriptors = None
previous_bboxes = []  # To store bounding boxes

# Open CSV file to log tracking data
with open('orb_vehicle_tracking_data.csv', mode='w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Frame', 'Vehicle_ID', 'X', 'Y', 'Width', 'Height', 'Matched_Keypoints'])

    frame_count = 0
    vehicle_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Preprocess frame for contour detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours to detect cars based on size and shape
        filtered_contours = []
        current_bboxes = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(w) / h

                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area > 0 else 0

                if min_aspect_ratio < aspect_ratio < max_aspect_ratio and solidity >= min_solidity:
                    current_bboxes.append((x, y, w, h))
                    filtered_contours.append(cnt)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Initialize ORB feature storage for the current frame
        current_keypoints = []
        current_descriptors = []

        # Apply ORB to each detected car's region
        for i, bbox in enumerate(current_bboxes):
            x, y, w, h = bbox
            car_region_gray = gray[y:y + h, x:x + w]  # Crop the car region
            keypoints, descriptors = orb.detectAndCompute(car_region_gray, None)

            # Store keypoints and descriptors for matching
            current_keypoints.append(keypoints)
            current_descriptors.append(descriptors)

            # Visualize keypoints on the frame
            for kp in keypoints:
                kp_x, kp_y = int(kp.pt[0]) + x, int(kp.pt[1]) + y
                cv2.circle(frame, (kp_x, kp_y), 3, (0, 255, 255), -1)

            # Save vehicle details to CSV
            csv_writer.writerow([frame_count, vehicle_id, x, y, w, h, len(keypoints)])
            vehicle_id += 1

        # Match features with the previous frame using BFMatcher
        # Match features with the previous frame using BFMatcher
        if previous_descriptors is not None:
            for i, descriptors in enumerate(current_descriptors):
                if descriptors is not None and previous_descriptors[i] is not None:
                    # Perform the matching between current and previous descriptors
                    matches = bf.match(previous_descriptors[i], descriptors)
                    matches = sorted(matches, key=lambda x: x.distance)
            
                    # Filter matches by distance threshold
                    threshold = 30  # Set the threshold to a reasonable value
                    good_matches = [m for m in matches if m.distance < threshold]
            
                    # Draw lines between matched keypoints for visualization
                    for match in good_matches:  
                        prev_kp = previous_keypoints[i][match.queryIdx].pt
                        curr_kp = keypoints[match.trainIdx].pt
                        prev_pt = (int(prev_kp[0]) + previous_bboxes[i][0], int(prev_kp[1]) + previous_bboxes[i][1])
                        curr_pt = (int(curr_kp[0]) + x, int(curr_kp[1]) + y)
                        cv2.line(frame, prev_pt, curr_pt, (255, 0, 0), 1)

                    # Save the number of good matches for logging or analysis
                    matched_keypoints_count = len(good_matches)
                    csv_writer.writerow([frame_count, vehicle_id, x, y, w, h, matched_keypoints_count])

        # Update previous frame data
        previous_bboxes = current_bboxes
        previous_keypoints = current_keypoints
        previous_descriptors = current_descriptors

        # Display the frame
        cv2.imshow('ORB Car Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
