import os
import random
import shutil

# 設定資料集路徑和分配比例
dataset_path = "/mnt/c/workspace/112-2/AI_Capstone/yolo/data"
train_ratio = 0.72
val_ratio = 0.14
test_ratio = 0.14

# 建立資料集的子資料夾
train_image_dir = os.path.join(dataset_path, "images/train")
val_image_dir = os.path.join(dataset_path, "images/val")
test_image_dir = os.path.join(dataset_path, "images/test")
train_label_dir = os.path.join(dataset_path, "labels/train")
val_label_dir = os.path.join(dataset_path, "labels/val")
test_label_dir = os.path.join(dataset_path, "labels/test")

# 刪除已存在的子資料夾及內容
shutil.rmtree(train_image_dir, ignore_errors=True)
shutil.rmtree(val_image_dir, ignore_errors=True)
shutil.rmtree(test_image_dir, ignore_errors=True)
shutil.rmtree(train_label_dir, ignore_errors=True)
shutil.rmtree(val_label_dir, ignore_errors=True)
shutil.rmtree(test_label_dir, ignore_errors=True)

# 重新建立空的子資料夾
os.makedirs(train_image_dir, exist_ok=True)
os.makedirs(val_image_dir, exist_ok=True)
os.makedirs(test_image_dir, exist_ok=True)
os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)
os.makedirs(test_label_dir, exist_ok=True)

# 獲取所有照片的檔案名
image_files = [f for f in os.listdir(os.path.join(dataset_path, "images")) if f.startswith("data") and f.endswith(".jpg")]

# 隨機打亂照片的順序
random.shuffle(image_files)

# 計算每個子集的照片數量
total_images = len(image_files)
train_count = int(total_images * train_ratio)
val_count = int(total_images * val_ratio)
test_count = total_images - train_count - val_count

# 分配照片到訓練集、驗證集和測試集
train_images = image_files[:train_count]
val_images = image_files[train_count:train_count+val_count]
test_images = image_files[train_count+val_count:]

# 將照片和標註檔案複製到對應的子資料夾
for image in train_images:
    src_image_path = os.path.join(dataset_path, "images", image)
    dst_image_path = os.path.join(train_image_dir, image)
    src_label_path = os.path.join(dataset_path, "labels", image.replace(".jpg", ".txt"))
    dst_label_path = os.path.join(train_label_dir, image.replace(".jpg", ".txt"))
    shutil.copy(src_image_path, dst_image_path)
    shutil.copy(src_label_path, dst_label_path)

for image in val_images:
    src_image_path = os.path.join(dataset_path, "images", image)
    dst_image_path = os.path.join(val_image_dir, image)
    src_label_path = os.path.join(dataset_path, "labels", image.replace(".jpg", ".txt"))
    dst_label_path = os.path.join(val_label_dir, image.replace(".jpg", ".txt"))
    shutil.copy(src_image_path, dst_image_path)
    shutil.copy(src_label_path, dst_label_path)

for image in test_images:
    src_image_path = os.path.join(dataset_path, "images", image)
    dst_image_path = os.path.join(test_image_dir, image)
    src_label_path = os.path.join(dataset_path, "labels", image.replace(".jpg", ".txt"))
    dst_label_path = os.path.join(test_label_dir, image.replace(".jpg", ".txt"))
    shutil.copy(src_image_path, dst_image_path)
    shutil.copy(src_label_path, dst_label_path)

print("分配完成！")
print(f"訓練集： {len(train_images)} 張照片")
print(f"驗證集： {len(val_images)} 張照片")
print(f"測試集： {len(test_images)} 張照片")