import cv2
import os

# 載入人臉檢測器
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

root_path = 'images'
num = 0
for root, dirs, files in os.walk(root_path):
    for file in files:
        img_path = os.path.join(root, file)
        # 讀取圖片
        image = cv2.imread(img_path)
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 檢測人臉
        faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # 開啟文本文件準備寫入
        with open('dataset.txt', 'a') as file:
            file.write('data{}.jpg {}\n'.format(num, len(faces)))
            i = 1
            cv2.imwrite('data{}.jpg'.format(num), image)
            # 遍歷每一個檢測到的人臉
            for (x, y, w, h) in faces:
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(image, str(i), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                # 將正規化後的人臉範圍 x_center、y_center、寬度和高度寫入文件
                x_center = (x + w / 2) / image.shape[1]
                y_center = (y + h / 2) / image.shape[0]
                w = w / image.shape[1]
                h = h / image.shape[0]
                file.write(f'0 {x_center} {y_center} {w} {h}\n')
                i += 1

            # image = cv2.resize(image, (0, 0), fx = 0.1, fy = 0.1)
            cv2.imshow('Image', image)
            # cv2.waitKey(0)
            cv2.imwrite('result{}.jpg'.format(num), image)
            print(f'檢測到 {len(faces)} 個人臉,已保存到 dataset.txt')
            num += 1


