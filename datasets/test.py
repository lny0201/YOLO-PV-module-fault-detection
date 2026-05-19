import torch
from ultralytics import YOLO
import os
import cv2
import numpy as np

# 定义输入文件夹和输出文件夹路径
input_folder = '/data4/zheli/yolov10-main/datasets/test/images'
output_folder = '/data4/zheli/yolov10-main/datasets/test/results'
labels_folder = '/data4/zheli/yolov10-main/datasets/test/labels'

# 加载三个模型
model1 = YOLO('/data4/zheli/yolov10-main/datasets/model/YOLOv8/best.pt')
model2 = YOLO('/data4/zheli/yolov10-main/datasets/model/YOLOv10/best.pt')
model3 = YOLO('/data4/zheli/yolov10-main/datasets/model/YOLOv11/best.pt')

# 获取输入文件夹中的图片文件列表
image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]


def similar_detection(detection1, detection2):
    x1_1, y1_1, x2_1, y2_1, label_1 = detection1
    x1_2, y1_2, x2_2, y2_2, label_2 = detection2
    center1 = ((x1_1 + x2_1) / 2, (y1_1 + y2_1) / 2)
    center2 = ((x1_2 + x2_2) / 2, (y1_2 + y2_2) / 2)
    distance = ((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2) ** 0.5
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    intersection_area = max(0, min(x2_1, x2_2) - max(x1_1, x1_2)) * max(0, min(y2_1, y2_2) - max(y1_1, y1_2))
    overlap = intersection_area / (area1 + area2 - intersection_area)
    return distance < 10 and overlap > 0.5 and label_1 == label_2


true_positives = 0
false_positives = 0
false_negatives = 0

for image_file in image_files:
    # 读取图片
    image_path = os.path.join(input_folder, image_file)
    img = cv2.imread(image_path)

    # 使用三个模型进行检测
    results1 = model1.predict(source=image_path, save=False)[0]
    results2 = model2.predict(source=image_path, save=False)[0]
    results3 = model3.predict(source=image_path, save=False)[0]

    # 存储所有模型的检测结果
    all_detections = []

    def process_results(results, model_name):
        for i, box in enumerate(results.boxes.xyxy):
            x1, y1, x2, y2 = box
            cls = results.boxes.cls[i]
            label = model_name.names[int(cls)]
            all_detections.append((x1.item(), y1.item(), x2.item(), y2.item(), label))

    process_results(results1, model1)
    process_results(results2, model2)
    process_results(results3, model3)

    # 处理重复检测结果并计算公共部分
    unique_detections = []
    for detection in all_detections:
        is_duplicate = False
        for unique_detection in unique_detections:
            if similar_detection(detection, unique_detection):
                is_duplicate = True
                break
        if not is_duplicate:
            unique_detections.append(detection)
        else:
            # 如果是重复检测，尝试计算公共部分
            x1s, y1s, x2s, y2s = [], [], [], []
            for results in [results1, results2, results3]:
                for i, box in enumerate(results.boxes.xyxy):
                    cls = results.boxes.cls[i]
                    if cls == detection[4]:
                        if similar_detection(detection, (box[0].item(), box[1].item(), box[2].item(), box[3].item(), detection[4])):
                            x1s.append(box[0].item())
                            y1s.append(box[1].item())
                            x2s.append(box[2].item())
                            y2s.append(box[3].item())
            if x1s:  # 检查 x1s 是否为空
                new_x1 = max(min(x1s), 0)
            else:
                new_x1 = detection[0]
            if y1s:
                new_y1 = max(min(y1s), 0)
            else:
                new_y1 = detection[1]
            if x2s:
                new_x2 = min(max(x2s), img.shape[1])
            else:
                new_x2 = detection[2]
            if y2s:
                new_y2 = min(max(y2s), img.shape[0])
            else:
                new_y2 = detection[3]
            # 更新重复检测结果为公共部分结果
            unique_detections[unique_detections.index(unique_detection)] = (new_x1, new_y1, new_x2, new_y2, detection[4])

    # 读取真实标签文件
    label_path = os.path.join(labels_folder, os.path.splitext(image_file)[0] + '.txt')
    true_labels = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                label = parts[0]
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                x1 = (x_center - width / 2) * img.shape[1]
                y1 = (y_center - height / 2) * img.shape[0]
                x2 = (x_center + width / 2) * img.shape[1]
                y2 = (y_center + height / 2) * img.shape[0]
                true_labels.append((x1, y1, x2, y2, label))

    # 计算准确率、召回率和F1分数相关变量
    for detection in unique_detections:
        found_match = False
        for true_label in true_labels:
            if similar_detection(detection, true_label):
                true_positives += 1
                found_match = True
                break
        if not found_match:
            false_positives += 1

    for true_label in true_labels:
        detected = False
        for detection in unique_detections:
            if similar_detection(detection, true_label):
                detected = True
                break
        if not detected:
            false_negatives += 1

    # 绘制检测结果
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 2
    for x1, y1, x2, y2, label in unique_detections:
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        text = f'{label}'
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        cv2.putText(img, text, (int(x1), int(y1) - 10), font, font_scale, (0, 255, 0), font_thickness)

    # 保存带有检测标记的图片到输出文件夹
    output_path = os.path.join(output_folder, image_file)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    cv2.imwrite(output_path, img)
    print("Saved image to:", output_path)

# 计算准确率、召回率和F1分数
if true_positives + false_positives > 0:
    precision = true_positives / (true_positives + false_positives)
else:
    precision = 0

if true_positives + false_negatives > 0:
    recall = true_positives / (true_positives + false_negatives)
else:
    recall = 0

if precision + recall > 0:
    f1_score = 2 * (precision * recall) / (precision + recall)
else:
    f1_score = 0

print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1_score}')