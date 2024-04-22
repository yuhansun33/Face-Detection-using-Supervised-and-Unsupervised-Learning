from skimage.feature import hog
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans
import os

def detect(kmeans, dataPath):
    with open(dataPath, "r") as file:
        line_list = [line.rstrip() for line in file]
    line_idx = 0

    while line_idx < len(line_list):
        tok = line_list[line_idx].split()
        img_name, num_faces = tok[0], int(tok[1])

        img = cv2.imread(os.path.join("detect", img_name))
        img_gray = cv2.imread(os.path.join("detect", img_name), cv2.IMREAD_GRAYSCALE)

        for i in range(num_faces):
            coord = [int(j) for j in line_list[line_idx + 1 + i].split()]
            img_face = img_gray[coord[1]:coord[1] + coord[3], coord[0]:coord[0] + coord[2]]
            # 對新圖片進行預處理
            new_image_resized = cv2.resize(img_face, (64, 64))
            new_hog_features = hog(new_image_resized, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))

            # 使用訓練好的K-means模型對新圖片進行預測
            new_pred_label = kmeans.predict([new_hog_features])[0]

            if new_pred_label == 0:
              color = (0, 255, 0)
            else:
              color = (0, 0, 255)
                
            cv2.rectangle(img, (coord[0], coord[1]), (coord[0] + coord[2], coord[1] + coord[3]), color, 2)
        cv2.imshow(img_name, img)
        # cv2.waitKey(0)
        cv2.imwrite(f'{img_name}', img)

        line_idx += (num_faces + 1)
