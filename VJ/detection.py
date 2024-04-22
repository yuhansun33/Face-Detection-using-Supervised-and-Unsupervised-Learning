import os
from unittest import result
import cv2
import utils
import numpy as np
import matplotlib.pyplot as plt


def detect(dataPath, clf):
    """
    Please read detectData.txt to understand the format. Load the image and get
    the face images. Transfer the face images to 19 x 19 and grayscale images.
    Use clf.classify() function to detect faces. Show face detection results.
    If the result is True, draw the green box on the image. Otherwise, draw
    the red box on the image.
      Parameters:A
        dataPath: the path of detectData.txt
      Returns:
        No returns.
    """
    # Begin your code (Part 4)
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
            img_face = cv2.resize(img_face, (19, 19))

            if clf.classify(img_face):
              color = (0, 255, 0)
            else:
              color = (0, 0, 255)
                
            cv2.rectangle(img, (coord[0], coord[1]), (coord[0] + coord[2], coord[1] + coord[3]), color, 2)
        cv2.imshow(img_name, img)
        # cv2.waitKey(0)
        cv2.imwrite(f'{img_name}', img)

        line_idx += (num_faces + 1)
        
    # End your code (Part 4)
