# API用モジュール

import cv2
import os
import numpy as np
from PIL import Image
import re
import os.path
import glob
from sklearn.externals import joblib


# フォルダ内の画像を習得
def get_images_and_labels():
    # トレーニング画像
    train_path = './train_乃木坂images'

    # Haar-like特徴分類器
    cascadePath = './haarcascade_frontalface_alt.xml'
    faceCascade = cv2.CascadeClassifier(cascadePath)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    # recognizer = cv2.createFisherFaceRecognizer()

    # 画像を格納する配列
    images = []
    # ラベルを格納する配列
    labels = []
    for f in os.listdir(train_path):
        # 画像のパス
        image_path = os.path.join(train_path, f)
        # 白黒で読み込み
        image_pil = Image.open(image_path).convert('L')
        # Numpyの配列に格納
        image = np.array(image_pil, 'uint8')
        # Haar-like特徴分類器で顔を検知
        faces = faceCascade.detectMultiScale(image)
        # 検出した画像の処理
        for(x, y, w, h) in faces:
            # 200×200にリサイズ
            roi = cv2.resize(image[y: y + h, x: x + w],
                             (200, 200), interpolation=cv2.INTER_LINEAR)
            # 画像を配列に格納
            images.append(roi)
            int_number = re.findall("\d+", f)
            for number in int_number:
                labels.append(int(number))
    # トレーニング実施
    model = recognizer.train(images, np.array(labels))

    # モデルの保存
    joblib.dump(model, './sample_model.pkl')


if __name__ == '__main__':
    get_images_and_labels()
