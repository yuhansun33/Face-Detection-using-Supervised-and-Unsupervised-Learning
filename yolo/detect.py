from ultralytics import YOLO
import cv2
import os

# 載入訓練好的模型
model = YOLO("best.pt")

file_number = 0
# 讀取照片
for root, dirs, files in os.walk("detect"):
    for file in files:
        img_path = os.path.join(root, file)
        img = cv2.imread(img_path)

        # 預測
        results = model.predict(source=img, save=True, save_txt=True, conf=0.5)

        # 取得預測結果圖片
        annotated_img = results[0].plot()

        # 顯示結果
        cv2.imshow("Face Detection", annotated_img)
        cv2.waitKey(0)

        # 儲存結果圖片
        save_path = "result_" + str(file_number) + ".jpg"
        cv2.imwrite(save_path, annotated_img)
        file_number += 1