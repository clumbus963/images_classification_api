from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import numpy as np
import cv2
from datetime import datetime
import os
import string
from PIL import Image


app = Flask(__name__)

def load_model():
    global recognizer
    print(" * Loading pre-trained model ...")
    cascadePath = './haarcascade_frontalface_alt.xml'
    faceCascade = cv2.CascadeClassifier(cascadePath)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    # recognizer = cv2.face.createLBPHFaceRecognizer()
    # recognizer = cv2.face.LBPHFaceRecognizer.create()
    recognizer.read('./sample_model.yml')

    print(' * Loading end')

@app.route('/')
def index():
    return render_template('./flask_api_index.html')

@app.route('/result', methods=['POST'])
def result():
    # submitした画像が存在したら処理する
    if request.files['image']:
        # 白黒画像として読み込み
        image_pil = Image.open(request.files['image']).convert('L')
        image = np.array(image_pil, 'uint8')
        # 類似度を出力
        label, predict_Confidence = recognizer.predict(image)
        predict_Confidence = str(predict_Confidence)
        # render_template('./result.html')
        return render_template('./result.html', title='類似度', predict_Confidence=predict_Confidence)

if __name__ == '__main__':
    load_model()
    app.debug = True
    app.run(host='localhost', port=5000)