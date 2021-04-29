import numpy as np
from PIL import Image
import os
import cv2
# 数据集路径
path = 'Facedata'

# 调用OpenCV_contrib库
recognizer = cv2.face.LBPHFaceRecognizer_create()
# 打开人脸分类器
detector = cv2.CascadeClassifier(r'F:\X41824141\venv\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')


def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]  # 拼接路劲
    faceSamples = []
    ids = []
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')   # convert it to grayscale
        img_numpy = np.array(PIL_img, 'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)
        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y + h, x: x + w])
            ids.append(id)
    return faceSamples, ids


# 利用OpenCV的训练函数进行训练
print('Now Loading...')
faces, ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))

# 保存训练信息
recognizer.write(r'face_trainer\trainer.yml')
print("{0} faces trained. Exiting Program".format(len(np.unique(ids))))