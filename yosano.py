#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
from PIL import Image


def facedetect(face_cascade, cap, image):
    global isFirst
    isFirst=True
    cnt = 0
    while(True):
        ret, frame = cap.read()

        if not ret:
            break
        else:
            if (cnt % 2) == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                facerect = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=2, minSize=(30, 30))

            cnt += 1

            if len(facerect) > 0:
                for rect in facerect:

                    x = rect[0]
                    y = rect[1]
                    w = rect[2]
                    h = rect[3]

                    # 顔の形状によっては調整が必要かも
                    x = x - w / 2
                    y = y - h / 1.8
                    w = w * 2
                    h = h * 2

                    x = int(round(x))
                    y = int(round(y))
                    w = int(round(w))
                    h = int(round(h))

                    # 矩形に合わせて合成する画像をリサイズ
                    image = cv2.resize(image, (w, h))
                    # カメラの顔を画像を合成
                    frame = overlay(frame, image, x, y)

                    # モザイク処理をしたい場合はこっち
                    # dst = frame[y:y+h, x:x+w]
                    # blur = cv2.blur(dst, (50, 50))
                    # frame[y:y+h, x:x+w] = blur
        imshow_fullscreen('result',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return 0


def overlay(frame, image, x, y):

    layer1 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    layer2 = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

    layer1 = Image.fromarray(layer1)
    layer2 = Image.fromarray(layer2)

    layer1 = layer1.convert('RGBA')
    layer2 = layer2.convert('RGBA')

    tmp = Image.new('RGBA', layer1.size, (255, 255, 255, 0))
    tmp.paste(layer2, (x, y), layer2)

    result = Image.alpha_composite(layer1, tmp)

    return cv2.cvtColor(np.asarray(result), cv2.COLOR_RGBA2BGRA)


def imshow_fullscreen(winname, img):
    cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
    # subprocess.call(["/usr/bin/osascript", "-e", 'tell app "Finder" to set frontmost of process "Python" to true'])
    cv2.setWindowProperty(winname, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(winname, img)


if __name__ == '__main__':
    # playSound()
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
    image = cv2.imread('yosano.png', cv2.IMREAD_UNCHANGED)
    ret, frame = cap.read()
    try:
        while True:
            k=cv2.waitKey(1)
            facedetect(face_cascade,cap,image)
            if k == 27:
                break      
    except KeyboardInterrupt:
        cap.release()
        cv2.destroyAllWindows()

    finally:
        cap.release()
        cv2.destroyAllWindows()
