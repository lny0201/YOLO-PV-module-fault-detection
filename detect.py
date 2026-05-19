import cv2
import supervision as sv
from ultralytics import YOLO
import os
import time

# 创建保存图片的目录
save_dir = r"D:\detect"
os.makedirs(save_dir, exist_ok=True)

model = YOLO('D:/yolov10-main/datasets/model/YOLOv8/best.pt')
bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("can't open the webcam")

# 初始化计数器和上次保存时间
detection_count = 0
last_save_time = 0
save_interval = 1  # 保存图片的最小间隔时间(秒)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    # 检查是否检测到目标
    if len(detections) > 0:
        current_time = time.time()
        # 控制保存频率，避免短时间内保存大量重复图片
        if current_time - last_save_time > save_interval:
            detection_count += 1
            # 生成带时间戳的文件名
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            save_path = os.path.join(save_dir, f"detection_{timestamp}_{detection_count}.jpg")
            # 保存带标注的图片
            annotated_frame = frame.copy()
            annotated_frame = bounding_box_annotator.annotate(scene=annotated_frame, detections=detections)
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)
            cv2.imwrite(save_path, annotated_frame)
            print(f"检测到目标，已保存图片: {save_path}")
            last_save_time = current_time
    # 实时显示带标注的画面
    annotated_image = bounding_box_annotator.annotate(scene=frame, detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)
    # 显示保存路径提示
    cv2.putText(annotated_image, f"Saved to: {save_dir}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow('Webcam', annotated_image)
    k = cv2.waitKey(1)
    if k % 256 == 27:
        print("Escape hit, closing...")
        break

cap.release()
cv2.destroyAllWindows()