import csv
import cv2
from ikomia.dataprocess.workflow import Workflow

# Video input and output paths
input_video_path = "cpy\car_in_dirt.mp4"
output_video_path = 'deepsort_output_video.avi'

# Initialize Ikomia workflow
wf = Workflow()
detector = wf.add_task(name="infer_yolo_v7", auto_connect=True)
tracking = wf.add_task(name="infer_deepsort", auto_connect=True)

# Set tracking parameters
tracking.set_parameters({
    "categories": "all",
    "conf_thres": "0.5",
})

clicked_point = None
selected_object_id = None

# Set up CSV file for logging
csv_file_path = 'selected_object_tracking_data.csv'
csv_file = open(csv_file_path, mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Frame Number', 'Object ID', 'Position X', 'Position Y'])  # Log frame number, object ID, and position

# Mouse click event to select an object for tracking
def click_event(event, x, y, flags, param):
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)
        print(f"Clicked at: {clicked_point}")

# Capture video input
stream = cv2.VideoCapture(input_video_path)
if not stream.isOpened():
    print("Error: Could not open video.")
    exit()

frame_width = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_rate = stream.get(cv2.CAP_PROP_FPS)

# Output video writer configuration
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (frame_width, frame_height))

cv2.namedWindow("DeepSORT")
cv2.setMouseCallback("DeepSORT", click_event)

frame_count = 0

# Define a display size (you can change the size depending on your screen resolution)
display_width = 1280
display_height = 1280

# Process each frame of the video
while True:
    ret, frame = stream.read()
    if not ret:
        print("End of video or error.")
        break

    frame_count += 1

    # Run the workflow on the frame but keep the original frame
    wf.run_on(array=frame)
    obj_detect_out = tracking.get_output(1)  # Get object detection output
    detections = obj_detect_out.get_objects()

    # Check if an object is clicked
    # <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>
    # edit here - remove to add yolov8 deepsort method - aerial view, train on VISdrone custom dataset 
    if clicked_point is not None:
        for detection in detections:
            bbox = detection.box
            x_min, y_min, width, height = bbox
            x_max = x_min + width
            y_max = y_min + height

            # Check if the click falls within a detection box
            if x_min <= clicked_point[0] <= x_max and y_min <= clicked_point[1] <= y_max:
                selected_object_id = detection.id  # Store selected object ID
                print(f"Selected object ID: {selected_object_id}")
                clicked_point = None  # Reset clicked point after object is selected
                break

    # If an object is selected, draw the bounding box around it on the original frame
    if selected_object_id is not None:
        selected_detection = next((d for d in detections if d.id == selected_object_id), None)

        if selected_detection:
            bbox = selected_detection.box
            x_min, y_min, width, height = bbox

            # Ensure bounding box coordinates are integers
            x_min = int(x_min)
            y_min = int(y_min)
            width = int(width)
            height = int(height)

            # Log Frame Number, Object ID, and Position (X, Y) to CSV
            csv_writer.writerow([frame_count, selected_object_id, x_min, y_min])

            # Draw bounding box around the selected object on the original frame
            cv2.rectangle(frame, (x_min, y_min), (x_min + width, y_min + height), (0, 255, 0), 3)  # Green box

    # Write the original frame (with bounding box) to the output video
    out.write(frame)

    # Resize frame for display only (this doesn't affect the saved video)
    display_frame = cv2.resize(frame, (display_width, display_height))

    # Display the resized frame
    cv2.imshow("DeepSORT", display_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
stream.release()
out.release()
csv_file.close()
cv2.destroyAllWindows()

'''


'''
