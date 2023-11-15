from ultralytics import YOLO
from SORT import *
import numpy as np
import datetime
import math
import cv2

model = YOLO(r"C:\Users\kosty\Downloads\weights.pt")  # Загрузка модели

cap = cv2.VideoCapture(0)  # Включение камеры

timer = [0.0 for _ in range(1000000)]  # Тестовая БД со временем падения людей

tracker = Sort(max_age=10, min_hits=3, iou_threshold=0.3)  # Настройка SORT

while cv2.waitKey(1) != ord("q"):
    flag, img = cap.read()  # Считывание изображения
    results = model.predict(img, stream=True, verbose=False)  # Получение результатов работы YoloV8
    detections = np.empty((0, 5))
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = math.ceil((box.conf[0] * 100)) / 100
            currentArray = np.array([x1, y1, x2, y2, conf])
            detections = np.vstack((detections, currentArray))
    resultsTracker = tracker.update(detections)

    for result in resultsTracker:
        x1, y1, x2, y2, Id = map(int, result)  # Получение Bounding Box'а и Id человека
        current = datetime.datetime.now().timestamp()  # Получение текущего времени
        if timer[Id] == 0.0:  # Если человек до этого не падал, установим ему время падения на текущее
            timer[Id] = current
        past = int(current - timer[Id])  # Определение сколько человек лежит
        color = (0, 255, 0)
        if past >= 10:  # Если человек лежит уже больше 10 секунд выделим его желтым
            color = (0, 255, 255)
        if past >= 20:  # Если человек лежит уже больше 20 секунд сообщаем оператору
            print(f"Человек лежит уже {past} секунд!!!")
            color = (0, 0, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.putText(img, str(past), (x1 + 5, y1 + 50), cv2.FONT_ITALIC, 2, (213, 155, 246), 5)  # Выделение человека

    cv2.imshow("Img", img)  # Отрисовка изображения
