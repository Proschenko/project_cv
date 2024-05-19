import cv2
import numpy as np

# Считывание изображения
image = cv2.imread('Dataset inner/Not_clean.png')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Бинаризация изображения
_, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

# Обнаружение краев
edges = cv2.Canny(binary, 50, 150, apertureSize=3)

# Преобразование Хафа для нахождения прямых линий
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)

# Установка порога длины линии
min_line_length = 150

# Фильтрация линий по длине
filtered_lines = []
for line in lines:
    for x1, y1, x2, y2 in line:
        line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if line_length > min_line_length:
            filtered_lines.append((x1, y1, x2, y2))

# Отображение отфильтрованных линий на изображении
output_image = image.copy()
for x1, y1, x2, y2 in filtered_lines:
    cv2.line(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow('Filtered Lines', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
