import cv2
import numpy as np

# Загрузка изображения
image_path = 'Dataset outter/Test15_out.jpg'
image = cv2.imread(image_path)

# Преобразование в оттенки серого
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Размытие
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Бинаризация
_, binary = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)

# Обнаружение границ
edges = cv2.Canny(binary, 50, 150)

# Применение морфологической операции закрытия для слияния близко расположенных границ
kernel = np.ones((5,5), np.uint8)
closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

# Применение скелетонизации для получения тонких линий
skel = np.zeros_like(closed_edges)
element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
done = False

while not done:
    eroded = cv2.erode(closed_edges, element)
    temp = cv2.dilate(eroded, element)
    temp = cv2.subtract(closed_edges, temp)
    skel = cv2.bitwise_or(skel, temp)
    closed_edges = eroded.copy()

    zeros = len(np.unique(closed_edges))
    if zeros == 1:
        done = True

# Создание вертикального структурного элемента
vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))

# Создание горизонтального структурного элемента
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))

# Нахождение вертикальных и горизонтальных линий
vertical_lines = cv2.morphologyEx(skel, cv2.MORPH_OPEN, vertical_kernel)
horizontal_lines = cv2.morphologyEx(skel, cv2.MORPH_OPEN, horizontal_kernel)

# Объединение вертикальных и горизонтальных линий
table_lines = cv2.bitwise_or(vertical_lines, horizontal_lines)

# Отображение промежуточных результатов
cv2.imshow('Оттенки серого', gray)
cv2.imshow('Размытие', blurred)
cv2.imshow('Бинаризация', binary)
cv2.imshow('Обнаружение границ', edges)
cv2.imshow('Закрытие границ', closed_edges)
cv2.imshow('Скелет', skel)
cv2.imshow('Вертикальные линии', vertical_lines)
cv2.imshow('Горизонтальные линии', horizontal_lines)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Отображение изображения с выделенными границами таблицы
cv2.imshow('Выделенные границы таблицы', table_lines)
cv2.waitKey(0)
cv2.destroyAllWindows()