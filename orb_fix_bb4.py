import cv2
import numpy as np

# cap = cv2.VideoCapture('Car-Tracking-Open-CV/videos/top_view_highway.mp4')
# cap = cv2.VideoCapture('Car-Tracking-Open-CV/videos/car_moving_away.mp4')
# cap = cv2.VideoCapture('Car-Tracking-Open-CV/videos/following_car.mp4')
cap = cv2.VideoCapture("cpy\car_in_dirt.mp4")

# Set target screen dimensions
screen_width, screen_height = 1920, 1080

roi_defined = False
roi = (0, 0, 0, 0)
template = None
tracking_threshold = 0.6
frame_count = 0
using_orb = False
using_optical_flow = False
using_template_matching = False
mask = None

orb = cv2.ORB_create(nfeatures=30)
lk_params = dict(winSize=(15, 15), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

def select_roi(event, x, y, flags, param):
    global roi_defined, roi, template, using_orb, using_optical_flow
    if event == cv2.EVENT_LBUTTONDOWN:
        roi = (x - 50, y - 50, 100, 100) 
        '''
        on selecting a different ROI, we are getting the problem of now enough keypoints are being detected in the new ROI.
        So the ROI size should be increased or decreased accordingly.
        '''
        roi_defined = True
        template = None
        using_orb = True 
        using_optical_flow = False
        using_template_matching = False
        print(f"ROI selected: {roi}")

cv2.namedWindow("Video")
cv2.setMouseCallback("Video", select_roi)

prev_gray = None
p0 = None  # Initial points for optical flow

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame to fit within screen dimensions while maintaining aspect ratio
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

    # Create a black canvas and place the resized frame in the center
    canvas = np.zeros((screen_height, screen_width, 3), dtype="uint8")
    y_offset = (screen_height - new_height) // 2
    x_offset = (screen_width - new_width) // 2
    canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_frame

    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

    if roi_defined:
        x, y, w, h = map(int, roi)

        if mask is None:
            mask = np.zeros((canvas.shape[0], canvas.shape[1]), dtype=np.uint8)
            mask[y:y+h, x:x+w] = 255

        if using_orb:
            roi_gray = gray[y:y+h, x:x+w]
            keypoints, descriptors = orb.detectAndCompute(roi_gray, None)
            for kp in keypoints:
                kp.pt = (kp.pt[0] + x, kp.pt[1] + y)

            # RANSAC Filtering: Filter out points that do not align with the main motion
            if len(keypoints) >= 4:  # Minimum required points for RANSAC
                pts1 = np.float32([kp.pt for kp in keypoints]).reshape(-1, 1, 2)
                pts2 = np.float32([kp.pt for kp in keypoints]).reshape(-1, 1, 2)

                # Find the homography using RANSAC
                H, mask_ransac = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
                inlier_pts = pts1[mask_ransac.ravel() == 1]

                # Filter out keypoints that are not inliers
                keypoints = [keypoints[i] for i in range(len(keypoints)) if mask_ransac[i]]

            frame_with_keypoints = cv2.drawKeypoints(canvas, keypoints, None, color=(0, 255, 0), flags=0)
            cv2.putText(frame_with_keypoints, "ORB tracking within ROI", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.rectangle(frame_with_keypoints, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imshow("Video", frame_with_keypoints)

            p0 = np.float32([kp.pt for kp in keypoints]).reshape(-1, 1, 2)
            prev_gray = gray.copy()
            using_orb = False
            using_optical_flow = True
            print("Switching to Optical Flow")

        elif using_optical_flow and p0 is not None:
            p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **lk_params)

            if p1 is not None and st is not None:
                good_new = p1[st == 1]
                good_old = p0[st == 1]

                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    cv2.line(canvas, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
                    cv2.circle(canvas, (int(a), int(b)), 5, (0, 0, 255), -1)

                x1, y1 = np.min(good_new, axis=0).ravel()
                x2, y2 = np.max(good_new, axis=0).ravel()
                cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                prev_gray = gray.copy()
                p0 = good_new.reshape(-1, 1, 2)

                if len(good_new) < 10:
                    print("Optical Flow lost track, switching to Template Matching")
                    using_optical_flow = False
                    using_template_matching = True
                    template = canvas[y:y+h, x:x+w]

                cv2.putText(canvas, "Optical Flow tracking", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                cv2.imshow("Video", canvas)

            else:
                using_optical_flow = False
                using_template_matching = True
                template = canvas[y:y+h, x:x+w]
                cv2.putText(canvas, "Optical Flow failed, switching to Template Matching", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.imshow("Video", canvas)

        elif using_template_matching and template is not None:
            if len(template.shape) == 3:
                template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

            mask = np.zeros(template.shape, dtype=np.uint8)
            mask[:, :] = 255

            res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED, mask=mask)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            
            if max_val >= tracking_threshold:
                top_left = max_loc
                bottom_right = (top_left[0] + w, top_left[1] + h)
                roi = (*top_left, w, h)

                mask = np.zeros((canvas.shape[0], canvas.shape[1]), dtype=np.uint8)
                mask[top_left[1]:top_left[1] + h, top_left[0]:top_left[0] + w] = 255

                cv2.rectangle(canvas, top_left, bottom_right, (0, 255, 0), 2)
                cv2.putText(canvas, f"Tracking object, score: {max_val:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                print("Template Matching failed, switching back to ORB")
                using_template_matching = False
                using_orb = True

            cv2.imshow("Video", canvas)

    else:
        cv2.putText(canvas, "Click to select ROI", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("Video", canvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
