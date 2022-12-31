# -*- coding: utf-8 -*-

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from yolo_model import model

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")

def load_dict_from_file(path_file):
    f = open(path_file,'r')
    data=f.read()
    f.close()
    return eval(data)

emotion_mapping = load_dict_from_file('dict.txt')
BGR_dict = load_dict_from_file('BGR_dict.txt')

# эта функция используется для задания внешнего вида надписи, отображающей распознанную эмоцию
def text_on_detected_boxes(text,text_x,text_y,image,font_scale = 1,
                           font = cv2.FONT_HERSHEY_COMPLEX_SMALL,
                           FONT_COLOR = (255, 255, 255),
                           FONT_THICKNESS = 1,
                           rectangle_bgr = (0, 255, 0)):
    # определяем ширину и высоту текстового поля
    (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]
    # установка координат рамок
    box_coords = ((text_x-10, text_y+4), (text_x + text_width+10, text_y - text_height-5))
    # отрисовка рамок и надписи
    cv2.rectangle(image, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
    cv2.putText(image, text, (text_x, text_y), font, fontScale=font_scale, color=FONT_COLOR,thickness=FONT_THICKNESS)

def face_detector_image(img):
    gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY) # преобразование в изображение в оттенках серого
    rgb_img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB) # преобразование в RGB-изображение
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.03, minNeighbors=10)
    if faces == ():
        return (0, 0, 0, 0), np.zeros((224, 224), np.uint8), img
    allfaces = []
    rects = []
    for (x, y, w, h) in faces:
        # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        roi_rgb = rgb_img[y:y + h, x:x + w]
        roi_rgb = cv2.resize(roi_rgb, (224, 224), interpolation=cv2.INTER_AREA)
        allfaces.append(roi_rgb)
        rects.append((x, w, y, h))
    return rects, allfaces, img

def emotionImage(imgPath):
    img = cv2.imread(imgPath)
    rects, faces, image = face_detector_image(img)
    i = 0
    for face in faces:
        roi = face.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        # прогноз эмоции
        label = model.predicted(roi, emotion_mapping)
        label_position = (rects[i][0] + int((rects[i][1] / 5)), abs(rects[i][2] - 10))
        label_color = BGR_dict[label]
        # наложение рамок на картинку
        x, w, y, h = rects[i]
        cv2.rectangle(image, (x, y), (x + w, y + h), label_color, 1)
        i += 1        
        # наложение обнаруженной эмоции на картинку    
        text_on_detected_boxes(label, label_position[0],label_position[1], image, rectangle_bgr = label_color)
    cv2.imshow('Emotion Detector', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def face_detector_video(img):
    gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY) # преобразование в изображение в оттенках серого
    rgb_img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB) # преобразование в RGB-изображение
    faces = face_classifier.detectMultiScale(gray, 1.1, 5)
    if faces == ():
        return (0, 0, 0, 0), np.zeros((224, 224), np.uint8), img
    allfaces = []
    rects = []
    for (x, y, w, h) in faces:
        # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        roi_rgb = rgb_img[y:y + h, x:x + w]
        roi_rgb = cv2.resize(roi_rgb, (224, 224), interpolation=cv2.INTER_AREA)
        allfaces.append(roi_rgb)
        rects.append((x, w, y, h))
    return rects, allfaces, img

def emotionVideo(cap):
    while True:
        ret, frame = cap.read()
        if ret:
            rects, faces, image = face_detector_video(frame)
            if np.sum(faces) != 0.0:
                i = 0
                for face in faces:
                    roi = face.astype("float") / 255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)
                    # прогноз эмоции
                    label = model.predicted(roi, emotion_mapping)
                    label_position = (rects[i][0] + int((rects[i][1] / 2)), abs(rects[i][2] - 10))
                    label_color = BGR_dict[label]
                    # наложение рамок на картинку
                    x, w, y, h = rects[i]
                    cv2.rectangle(image, (x, y), (x + w, y + h), label_color, 1)
                    i += 1
                    # наложение обнаруженной эмоции на картинку
                    text_on_detected_boxes(label, label_position[0],label_position[1], image, rectangle_bgr = label_color)
                fps = cap.get(cv2.CAP_PROP_FPS)
                cv2.putText(image, str(fps),(5, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)
            else:
                cv2.putText(image, "No Face Found", (5, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 1)
            cv2.imshow('All', image)
#             out.write(image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
          break
    cap.release()
#     out.release()
    cv2.destroyAllWindows()
    
