import cv2
import numpy as np

# Initialize video capture
cap = cv2.VideoCapture('Car-Tracking-Open-CV/videos/top_view_highway.mp4')

# ORB detector and Brute-Force Matcher setup
orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Parameters for contour filtering
min_area = 500
max_area = 3000
min_aspect_ratio = 0.5
max_aspect_ratio = 4.0
min_solidity = 0.7

# Store previous frame's data
previous_keypoints = []
previous_descriptors = None # just intiialise 
previous_bboxes = []  # To store bounding boxes

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

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
    for bbox in current_bboxes:
        x, y, w, h = bbox
        car_region_gray = gray[y:y + h, x:x + w]  # Crop the car region
        keypoints, descriptors = orb.detectAndCompute(car_region_gray, None)

        # Store keypoints and descriptors for matching
        current_keypoints.append(keypoints)
        current_descriptors.append(descriptors)

        # Visualize keypoints on the frame
        for kp in keypoints:
            kp_x, kp_y = int(kp.pt[0]) + x, int(kp.pt[1]) + y  # Adjust to full frame coordinates
            cv2.circle(frame, (kp_x, kp_y), 3, (0, 255, 255), -1)

    # Match features with the previous frame using BFMatcher
    if previous_descriptors is not None:
        for i, descriptors in enumerate(current_descriptors):
            if descriptors is not None and previous_descriptors[i] is not None:
                matches = bf.match(previous_descriptors[i], descriptors)
                matches = sorted(matches, key=lambda x: x.distance)  # Sort matches by distance

                # Draw lines between matched keypoints for visualization
                for match in matches[:10]:  # Limit the number of matches drawn
                    prev_kp = previous_keypoints[i][match.queryIdx].pt
                    curr_kp = keypoints[match.trainIdx].pt
                    prev_pt = (int(prev_kp[0]) + previous_bboxes[i][0], int(prev_kp[1]) + previous_bboxes[i][1])
                    curr_pt = (int(curr_kp[0]) + x, int(curr_kp[1]) + y)
                    cv2.line(frame, prev_pt, curr_pt, (255, 0, 0), 1)

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
