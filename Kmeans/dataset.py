import os
import cv2
import glob
import numpy as np
from skimage.feature import hog

def create_dataset():
    # 載入標註資料
    with open("dataset.txt") as file:
        line_list = [line.rstrip() for line in file]

    np.random.seed(0)

    dataset, label = [], []
    line_idx = 0

    while line_idx < len(line_list):
        tok = line_list[line_idx].split()
        img_gray = cv2.imread(os.path.join("images", tok[0] ), cv2.IMREAD_GRAYSCALE)
        num_faces = int(tok[1])

        # 裁切出人臉區域
        face_box_list = []
        for i in range(num_faces):
            #轉換座標
            coord = [float(j) for j in line_list[line_idx + 1 + i].split()]
            w = int(coord[3] * img_gray.shape[1])
            h = int(coord[4] * img_gray.shape[0])
            x = int(coord[1] * img_gray.shape[1] - w / 2)
            y = int(coord[2] * img_gray.shape[0] - h / 2)

            left_top = (max(x, 0), max(y, 0))
            right_bottom = (min(x + w, img_gray.shape[1]), min(y + h, img_gray.shape[0]))
            face_box_list.append([left_top, right_bottom])
            # cv2.rectangle(img_gray, left_top, right_bottom, (0, 255, 0), 2)

            img_crop = img_gray[left_top[1]:right_bottom[1], left_top[0]:right_bottom[0]].copy()
            # HOG
            hog_features = HOG(img_crop)
            # append to dataset
            dataset.append(hog_features)
            # dataset.append(img_flatten)
            label.append(1)

        line_idx += num_faces + 1

        # Random crop N non-face region
        # Here we set N equal to the number of faces to generate a balanced dataset
        # Note that we have alreadly save the bounding box of faces into `face_box_list`, you can utilize it for non-face region cropping
        for i in range(num_faces):
            o_x, o_y = face_box_list[i][0]
            w, h = face_box_list[i][1][0] - o_x, face_box_list[i][1][1] - o_y
            left_top, right_bottom = get_new_coord(o_x, o_y, w, h, img_gray.shape[1], img_gray.shape[0])
            img_crop = img_gray[left_top[1]:right_bottom[1], left_top[0]:right_bottom[0]].copy()

            # img_resize = cv2.resize(img_crop, (19, 19))
            # img_flatten = img_resize.flatten()
            # HOG
            hog_features = HOG(img_crop)
            # append to dataset
            dataset.append(hog_features)
            # dataset.append(img_flatten)
            label.append(0)

        # cv2.imshow("windows", img_gray)
        # cv2.waitKey(0)


    return dataset, label

def HOG(img):
    img_resize = cv2.resize(img, (64, 64))  # 調整圖片大小為64x64
    hog_features = hog(img_resize, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
    return hog_features

def get_new_coord(o_x, o_y, w, h, img_w, img_h):
    min_overlap_ratio = 0.1
    max_overlap_ratio = 0.3

    while True:
        offset_w = np.random.uniform(-w, w)
        offset_h = np.random.uniform(-h, h)

        left_top = (int(max(o_x + offset_w, 0)), int(max(o_y + offset_h, 0)))
        right_bottom = (int(min(left_top[0] + w, img_w)), int(min(left_top[1] + h, img_h)))
        

        overlap_ratio = ((right_bottom[0] - o_x) if left_top[0] > o_x else (o_x + w - left_top[0]) ) \
                        * ((right_bottom[1] - o_y) if left_top[1] < o_y else (o_y +h  - left_top[1])) \
                        / (w * h)
        if overlap_ratio > min_overlap_ratio and overlap_ratio < max_overlap_ratio:
            break

    return left_top, right_bottom


