from ultralytics import YOLO

def main():
    # 加载预训练模型（推荐用于训练）
    model = YOLO('yolov8n.pt')

    # 进行模型训练
    results = model.train(data='E:/Safety-Helmet-Wearing-Dataset/VOC2028/VOC2028.yaml', epochs=3)

    # 可以添加更多处理步骤

if __name__ == '__main__':
    main()
