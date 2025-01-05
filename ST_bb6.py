import cv2
import numpy as np

# Initialize video capture
cap = cv2.VideoCapture('Car-Tracking-Open-CV/videos/top_view_highway.mp4')

# Parameters
roi_defined = False
roi = (0, 0, 0, 0)  # Initial ROI
template = None
template_matching_method = cv2.TM_CCOEFF_NORMED
tracking_threshold = 0.6  # Correlation threshold to consider a match
frame_count = 0  # Counter to update template every few frames
mask = None  # Initial mask

def select_roi(event, x, y, flags, param):
    global roi_defined, roi, template, mask
    if event == cv2.EVENT_LBUTTONDOWN:
        roi = (x - 50, y - 50, 100, 100)  # Adjust ROI size if necessary
        roi_defined = True
        print(f"ROI selected: {roi}")
        mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)  # Initialize mask with zeros
        mask[y-50:y+50, x-50:x+50] = 255  # Set mask over ROI
        print("Mask initialized")

cv2.namedWindow("Video")
cv2.setMouseCallback("Video", select_roi)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if roi_defined:
        x, y, w, h = map(int, roi)

        # Initialize template if not already done
        if template is None:
            template = gray[y:y + h, x:x + w]
            print("Template initialized for tracking")

        # Ensure mask size matches template size
        mask_resized = mask[y:y + h, x:x + w]  # Crop mask to match template size

        # Perform template matching to locate the object with the mask
        res = cv2.matchTemplate(gray, template, template_matching_method, mask=mask_resized)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        if max_val >= tracking_threshold:
            top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)
            roi = (*top_left, w, h)  # Update ROI to match the object’s new position

            # Update mask to follow the new position of the template
            mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
            mask[top_left[1]:top_left[1] + h, top_left[0]:top_left[0] + w] = 255  # Set mask over updated ROI

            # Draw the bounding box around the detected area (the object)
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
            cv2.putText(frame, f"Tracking object, score: {max_val:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        else:
            # If no good match found, show tracking loss message
            cv2.putText(frame, "Lost track - no good match", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Periodically update the template if the object moves significantly
        if frame_count % 30 == 0:  # Update template every 30 frames, for example
            template = gray[y:y + h, x:x + w]  # Reinitialize the template
            mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
            mask[y:y + h, x:x + w] = 255  # Update the mask to match the new template
            print("Template and mask updated for tracking")

        frame_count += 1

    else:
        cv2.putText(frame, "Click to select ROI", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Display the result
    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
