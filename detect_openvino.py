from ultralytics import YOLO
import cv2
import ipywidgets as widgets
import openvino as ov
from pathlib import Path
from openvino.runtime import Core, Model
import time
import os

# 获取图片路径列表
dataset_path = r"E:\Safety-Helmet-Wearing-Dataset\bicycle\merged_dataset\images\train"
test_images = [os.path.join(dataset_path, img) for img in os.listdir(dataset_path) if img.endswith(('.png', '.jpg', '.jpeg'))]
length = len(test_images)

core = Core()
# 电动车检测openvino路径
model_path = Path("bicycle_weights/best_openvino_model/best.xml")
# model = YOLO("bicycle_weights/best_openvino_model/")
# 电动车检测模型
model = YOLO(r"bicycle_weights\best.pt")  # load a custom model
ov_model = core.read_model(model_path)
compiled_ov_model = core.compile_model(ov_model, "CPU")
# if not model_path.exists():
# 	model.export(format="openvino", dynamic=True, half=False)


# 头盔检查模型
model_path_helmet = Path("runs/detect/train7/weights/best_openvino_model/best.xml")
# model_helmet = YOLO("runs/detect/train7/weights/best_openvino_model/")
model_helmet = YOLO(r"runs\detect\train7\weights\best.pt")
ov_model_helmet = core.read_model(model_path_helmet)
compiled_ov_model_helmet = core.compile_model(ov_model_helmet, "CPU")
# if not model_path_helmet.exists():
# 	model_helmet.export(format="openvino", dynamic=True, half=False)

import torch


def infer(*args):
    result = compiled_ov_model(args)[0]
    return torch.from_numpy(result)


model.predictor.inference = infer


def infer_(*args):
    result = compiled_ov_model_helmet(args)[0]
    return torch.from_numpy(result)


model_helmet.predictor.inference = infer_

start = time.time()
# 测试图片集
# test_images = [r"test_image_1.jpg", r"test_image_2.png"]
# length = len(test_images)
results = model(test_images, device="cpu")  # return a list of Results objects

# 函数用于检测头盔, 并修改图片
def detect_helmet(cropped_image):
    # 假设model_helmet是已经加载的模型，并对输入的图像进行头盔检测
    result = model_helmet(cropped_image, device="cpu")
    boxes = result[0].boxes
    helmet_detections = []

    # 遍历每个检测到的边界框
    for i, bbox in enumerate(boxes.xyxy):
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])  # 从bbox提取坐标
        # 获取类别，这里假设cls为0表示佩戴了头盔
        if int(boxes.cls[i]) == 0:
            label = 'Helmet'
            color = (0, 255, 0)  # 绿色表示佩戴了头盔
            helmet_detections.append({
                'bbox': bbox.tolist(),  # 将tensor转换为list
                'label': 'helmet'       # 标记为佩戴头盔
            })
        else:
            label = 'No Helmet'
            color = (0, 0, 255)  # 红色表示未佩戴头盔
            helmet_detections.append({
                'bbox': bbox.tolist(),  # 将tensor转换为list
                'label': 'no helmet'    # 标记为未佩戴头盔
            })
        # 在图像上绘制边界框和标签
        cv2.rectangle(cropped_image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(cropped_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return helmet_detections

# 用于存储最终结果
final_results = []

# Process results list
for i in range(length):
    original_image = cv2.imread(test_images[i])
    boxes = results[i].boxes  # Boxes object for bounding box outputs
    # print("boxes:", boxes)
    # print(boxes.cls)
    for t, bbox in enumerate(boxes.xyxy):
        if int(boxes.cls[t]) == 4: # 筛选cls为电瓶车的结果
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            box_height = y2 - y1
            # 调整y1以包括等于框高度的上方区域
            y1 = max(0, y1 - box_height)  # 防止越界
            # 裁剪图像
            cropped_image = original_image[y1:y2, x1:x2]
            # 检测头盔
            helmet_detections = detect_helmet(cropped_image)
            # 保存结果
            final_results.append({
                'scooter_bbox': (x1, y1, x2, y2),
                'helmet_detections': helmet_detections
            })
    # cv2.imwrite(f'annotated_image_{i}.jpg', original_image)

# 计算FPS
end = time.time()
total_time = end - start
fps = length / total_time
print(f"FPS: {fps:.2f}")
print("总时长", total_time)

# # 输出结果
# for result in final_results:
#     print(result)

# print("总时长", time.time()-start)
