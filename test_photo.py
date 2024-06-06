from ultralytics import YOLO
import cv2
import time
import os


test_images = ["test.jpg", "test_image_1.jpg", "test_image_2.png"]
length = len(test_images)

# start = time.time()
# 电动车检测模型
model = YOLO(r"bicycle_weights\best.pt")  # load a custom model
# # 测试图片集
# test_images = [r"test_image_1.jpg",
#                  r"test_image_2.png"]
# length = len(test_images)
start = time.time()
results = model(test_images)  # return a list of Results objects
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename=f"电动车检测.jpg")  # save to disk


# 头盔检查模型
model_helmet = YOLO(r"runs\detect\train7\weights\best.pt")
temps = model_helmet(test_images)
for result in temps:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename=f"头盔检测.jpg")  # save to disk

# 函数用于检测头盔, 并修改图片
def detect_helmet(cropped_image):
    # 假设model_helmet是已经加载的模型，并对输入的图像进行头盔检测
    result = model_helmet(cropped_image)
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
    cv2.imwrite(f'annotated_image_{i}.jpg', original_image)

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
