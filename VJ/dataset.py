import os
import cv2
import glob
import numpy as np

def create_dataset():
    """
        This function generates the training and testing dataset  form the path: 'data/data_small'.
        The dataset is a list of tuples where the first element is the numpy array of shape (m, n)
        representing the image the second element is its classification (1 or 0).
        
        In the following, there are 4 main steps:
        1. Read the .txt file
        2. Crop the faces using the ground truth label in the .txt file
        3. Random crop the non-faces region
        4. Split the dataset into training dataset and testing dataset
        
        Parameters:
            data_idx: the data index string of the .txt file

        Returns:
            train_dataset: the training dataset
            test_dataset: the testing dataset
    """

    with open("dataset.txt") as file:
        line_list = [line.rstrip() for line in file]

    # Set random seed for reproducing same image croping results
    np.random.seed(0)

    face_dataset, nonface_dataset = [], []
    line_idx = 0

    # Iterate through the .txt file
    # The detail .txt file structure can be seen in the README at https://vis-www.cs.umass.edu/fddb/
    while line_idx < len(line_list):
        tok = line_list[line_idx].split()
        img_gray = cv2.imread(os.path.join("images", tok[0] ), cv2.IMREAD_GRAYSCALE)
        num_faces = int(tok[1])

        # Crop face region using the ground truth label
        face_box_list = []
        for i in range(num_faces):
            # Here, each face is denoted by:
            # <major_axis_radius minor_axis_radius angle center_x center_y 1>.
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
            face_dataset.append((cv2.resize(img_crop, (19, 19)), 1))

        line_idx += num_faces + 1

        # Random crop N non-face region
        # Here we set N equal to the number of faces to generate a balanced dataset
        # Note that we have alreadly save the bounding box of faces into `face_box_list`, you can utilize it for non-face region cropping
        for i in range(num_faces):
            # Begin your code (Part 1-2)
            o_x, o_y = face_box_list[i][0]
            w, h = face_box_list[i][1][0] - o_x, face_box_list[i][1][1] - o_y
            left_top, right_bottom = get_new_coord(o_x, o_y, w, h, img_gray.shape[1], img_gray.shape[0])
            img_crop = img_gray[left_top[1]:right_bottom[1], left_top[0]:right_bottom[0]].copy()
            # End your code (Part 1-2)

            nonface_dataset.append((cv2.resize(img_crop, (19, 19)), 0))

        # cv2.imshow("windows", img_gray)
        # cv2.waitKey(0)

    # train test split
    num_face_data, num_nonface_data = len(face_dataset), len(nonface_dataset)
    SPLIT_RATIO = 0.7

    train_dataset = face_dataset[:int(SPLIT_RATIO * num_face_data)] + nonface_dataset[:int(SPLIT_RATIO * num_nonface_data)]
    test_dataset = face_dataset[int(SPLIT_RATIO * num_face_data):] + nonface_dataset[int(SPLIT_RATIO * num_nonface_data):]

    return train_dataset, test_dataset

def get_new_coord(o_x, o_y, w, h, img_w, img_h):
    min_overlap_ratio = 0.1
    max_overlap_ratio = 0.3
    with open('experiment.txt', 'a') as f:
        f.write('(overlap ratio {}-{})\n'.format(min_overlap_ratio, max_overlap_ratio))

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


