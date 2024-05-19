import cv2
import numpy as np

# Шаг 1: Загрузка и предобработка изображения
image = cv2.imread('Dataset inner/Not_clean.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

# Шаг 2: Обнаружение границ таблицы
edges = cv2.Canny(binary, 50, 150, apertureSize=3)
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

# Шаг 3: Извлечение ячеек таблицы
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Применяем морфологические операции для выделения ячеек
kernel = np.ones((5, 5), np.uint8)
dilated = cv2.dilate(edges, kernel, iterations=1)

contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
some_threshold = 100
# # Шаг 4: Разделение фамилий и подписей
# for contour in contours:
#     x, y, w, h = cv2.boundingRect(contour)
#     if w > 50 and h > 20:  # Фильтрация слишком маленьких областей
#         segment = image[y:y + h, x:x + w]
#
#         # Определяем, является ли сегмент фамилией или подписью по координатам
#         if y < some_threshold:  # Установите порог для разделения фамилий и подписей
#             cv2.imwrite(f'surnames/surname_{x}_{y}.png', segment)
#         else:
#             cv2.imwrite(f'signatures/signature_{x}_{y}.png', segment)

# Сохраняем результат
cv2.imwrite('result_image.jpg', image)
