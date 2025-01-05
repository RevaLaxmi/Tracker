import cv2
import numpy as np

# cap = cv2.VideoCapture('Car-Tracking-Open-CV/videos/top_view_highway.mp4')
# cap = cv2.VideoCapture('Car-Tracking-Open-CV/videos/car_moving_away.mp4')
''' check ^ this - run and fix the error'''
# cap = cv2.VideoCapture('Car-Tracking-Open-CV/videos/following_car.mp4')
# cap = cv2.VideoCapture('Car-Tracking-Open-CV/videos/car_reduction.mp4')
''' check ^ this - run and fix the error'''
# cap = cv2.VideoCapture('Car-Tracking-Open-CV/videos/car_in_dirt.mp4')
# cap = cv2.VideoCapture('Car-Tracking-Open-CV/videos/bike_on_hill2.mp4')
# cap = cv2.VideoCapture('Car-Tracking-Open-CV/videos/caronhill_topview.mp4')
# cap = cv2.VideoCapture('Car-Tracking-Open-CV/videos/vehiclesonhill_topview.mp4')
''' check ^ this - run and fix the error - same double click error'''
# cap = cv2.VideoCapture('Car-Tracking-Open-CV/videos/top_view_crossway1.mp4')
cap = cv2.VideoCapture('Car-Tracking-Open-CV/videos/topview_roundaboutsmallv.mp4')

screen_width, screen_height = 1920, 1080
roi_defined = False
roi = (0, 0, 0, 0)

orb = cv2.ORB_create(nfeatures=500)
lk_params = dict(winSize=(15, 15), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

def select_roi(event, x, y, flags, param):
    global roi_defined, roi, keypoints, p0, prev_gray
    if event == cv2.EVENT_LBUTTONDOWN:
        roi = (x - 40 - x_offset, y - 40 - y_offset, 80, 80)  # Fixed 80x80 ROI
        roi_defined = True
        keypoints = None  # Reset keypoints
        p0 = None
        prev_gray = None
        print(f"ROI selected: {roi}")

cv2.namedWindow("Video")
cv2.setMouseCallback("Video", select_roi)

prev_gray = None
p0 = None  # Initial points for optical flow

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame to fit within screen dimensions
    frame_height, frame_width = frame.shape[:2]
    frame_aspect_ratio = frame_width / frame_height
    screen_aspect_ratio = screen_width / screen_height

    if frame_aspect_ratio > screen_aspect_ratio:
        new_width = screen_width
        new_height = int(screen_width / frame_aspect_ratio)
    else:
        new_height = screen_height
        new_width = int(screen_height * frame_aspect_ratio)

    resized_frame = cv2.resize(frame, (new_width, new_height))
    canvas = np.zeros((screen_height, screen_width, 3), dtype="uint8")
    y_offset = (screen_height - new_height) // 2
    x_offset = (screen_width - new_width) // 2
    canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_frame

    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

    if roi_defined:
        x, y, w, h = map(int, roi)

        if p0 is None:
            # Initialize ORB keypoints within the selected ROI
            roi_gray = gray[y:y + h, x:x + w]
            keypoints = orb.detect(roi_gray, None)
            if len(keypoints) > 0:
                p0 = np.float32([(kp.pt[0] + x, kp.pt[1] + y) for kp in keypoints]).reshape(-1, 1, 2)
                prev_gray = gray.copy()
        else:
            # Optical Flow to track keypoints
            p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **lk_params)
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            # Ensure keypoints stay close to their original positions
            distances = np.linalg.norm(good_new - good_old, axis=1)
            max_distance = 30  # Adjust this threshold as needed
            valid_points = distances < max_distance
            good_new = good_new[valid_points]

            # Recalculate ROI based on the mean position of keypoints
            if len(good_new) > 0:
                x_center = int(np.mean(good_new[:, 0]))
                y_center = int(np.mean(good_new[:, 1]))

                # Keep the ROI fixed in size but centered around keypoints
                roi = (
                    max(0, x_center - w // 2),
                    max(0, y_center - h // 2),
                    w,
                    h
                )

                # Update keypoints for the next frame
                p0 = good_new.reshape(-1, 1, 2)
                prev_gray = gray.copy()

        # Draw the fixed ROI and keypoints
        cv2.rectangle(canvas, (roi[0], roi[1]), (roi[0] + roi[2], roi[1] + roi[3]), (0, 255, 0), 2)
        for pt in p0:
            cv2.circle(canvas, (int(pt[0][0]), int(pt[0][1])), 3, (0, 0, 255), -1)

    # Display instructions or tracking status
    if not roi_defined:
        cv2.putText(canvas, "Click to select ROI", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    else:
        cv2.putText(canvas, "Tracking object", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Video", canvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
