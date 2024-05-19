import cv2

# Загрузка изображения
image_path = 'Dataset inner/Clean_table.png'
image = cv2.imread(image_path)

# Преобразование в оттенки серого
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Размытие
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Бинаризация
_, binary = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)

# Обнаружение границ
edges = cv2.Canny(binary, 50, 150)

# Поиск контуров
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Отображение промежуточных результатов
cv2.imshow('Оттенки серого', gray)
cv2.imshow('Размытие', blurred)
cv2.imshow('Бинаризация', binary)
cv2.imshow('Обнаружение границ', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Отрисовка контуров на исходном изображении
image_with_contours = image.copy()
cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)

# Отображение изображения с выделенными контурами
cv2.imshow('Выделенные границы таблицы', image_with_contours)
cv2.waitKey(0)
cv2.destroyAllWindows()


























