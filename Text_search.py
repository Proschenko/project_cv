import cv2
import pytesseract

# Загрузка изображения
image_path = 'Dataset inner/Text.png'
image = cv2.imread(image_path)

# Преобразование изображения в оттенки серого
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Применение порогового значения для бинаризации изображения
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# Применение операции расширения для улучшения качества текста
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
dilated = cv2.dilate(binary, kernel, iterations=1)

# Распознавание текста с помощью Tesseract
text = pytesseract.image_to_string(dilated, lang='rus')

# Вывод распознанного текста
print("Распознанный текст:")
print(text)
