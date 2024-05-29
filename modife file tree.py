import os
import shutil

# 设置基本路径和目标目录
base_path = r'E:\Safety-Helmet-Wearing-Dataset\VOC2028'
txt_files = {
    'train': 'train.txt',
    'val': 'val.txt',
    'test': 'test.txt'
}
# image_base_folder = os.path.join(base_path, 'images')
labels_base_folder = os.path.join(base_path, 'labels')
# dest_folders = {
#     'train': os.path.join(base_path, 'images', 'train'),
#     'val': os.path.join(base_path, 'images', 'val'),
#     'test': os.path.join(base_path, 'images', 'test')
# }
dest_labels_folders = {
    'train': os.path.join(base_path, 'labels', 'train'),
    'val': os.path.join(base_path, 'labels', 'val'),
    'test': os.path.join(base_path, 'labels', 'test')
}

# 确保目标文件夹存在
# for folder in dest_folders.values():
#     os.makedirs(folder, exist_ok=True)

for folder in dest_labels_folders.values():
    os.makedirs(folder, exist_ok=True)


# 读取每个txt文件，并移动图像到相应的目标文件夹
for key, txt_file in txt_files.items():
    with open(os.path.join(base_path, txt_file), 'r') as file:
        image_paths = file.readlines()
        for image_path in image_paths:
            image_path = image_path.strip()  # 移除可能的换行符
            file_name = os.path.basename(image_path)  # 获取文件名
            label_file_name = os.path.splitext(file_name)[0] + '.txt'
            src_label_path = os.path.join(labels_base_folder, label_file_name)
            dest_label_path = os.path.join(dest_labels_folders[key], label_file_name)

            # if os.path.exists(image_path):  # 确保源文件存在
            #     file_name = os.path.basename(image_path)  # 获取文件名
            #     dest_image_path = os.path.join(dest_folders[key], file_name)
            #     shutil.move(image_path, dest_image_path)
            # else:
            #     print(f"Warning: {image_path} does not exist and cannot be moved.")

            # 移动标签文件
            if os.path.exists(src_label_path):
                shutil.move(src_label_path, dest_label_path)
            else:
                print(f"Warning: {src_label_path} does not exist and cannot be moved.")

print("Images have been successfully reorganized into test, train, and val folders.")
