# from inference import InferencePipeline
# from inference.core.interfaces.stream.sinks import render_boxes
 
# pipeline = InferencePipeline.init(
#     model_id="yolov12n-640",
#     video_reference=0,
#     on_prediction=render_boxes
# )
# pipeline.start()
# pipeline.join()


from ultralytics import YOLO
import cv2

# Load the YOLOv8n model
model = YOLO("yolov8n.pt")  # or use yolov8s.pt, yolov8m.pt for higher accuracy

# Load the video file
video_path = "traffic.mp4"  # Replace with your video file path
cap = cv2.VideoCapture(video_path)

# Get video writer setup
output_path = "output.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference on the frame
    results = model(frame)

    # Draw the results on the frame
    annotated_frame = results[0].plot()

    # Show the frame (optional)
    cv2.imshow("YOLOv8 Inference", annotated_frame)

    # Save the frame
    out.write(annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()