import cv2
import numpy as np

def preprocess_image(image_path):
    # Чтение изображения
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Не удается открыть файл: {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, gray

def detect_table(gray, blur_ksize, threshold_type, block_size):
    # Применение гауссового размытия для уменьшения шумов
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    # Применение адаптивного порогового значения
    if threshold_type == 'adaptive':
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, 2)
    elif threshold_type == 'binary':
        _, thresh = cv2.threshold(blurred, block_size, 255, cv2.THRESH_BINARY_INV)

    # Применение морфологических операций для улучшения границ таблицы
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Поиск контуров
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def extract_table(image, contours):
    if not contours:
        return None

    # Поиск наибольшего контура
    max_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(max_contour, True)
    approx = cv2.approxPolyDP(max_contour, epsilon, True)

    if len(approx) == 4:
        # Найти углы и применить перспективное преобразование
        pts = np.array([pt[0] for pt in approx], dtype='float32')
        rect = order_points(pts)
        (tl, tr, br, bl) = rect

        # Вычисление ширины и высоты нового изображения
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = max(int(heightA), int(heightB))

        # Определение конечных точек
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        # Применение матрицы перспективного преобразования
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

        return warped
    else:
        return None

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def save_image(image, output_path):
    cv2.imwrite(output_path, image)

def main(image_path, output_path):
    try:
        image, gray = preprocess_image(image_path)
        attempts = 10
        blur_ksize_list = [5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
        block_size_list = [11, 13, 15, 17, 19, 21, 23, 25, 27, 29]
        threshold_types = ['adaptive', 'binary']

        for attempt in range(attempts):
            blur_ksize = blur_ksize_list[attempt % len(blur_ksize_list)]
            block_size = block_size_list[attempt % len(block_size_list)]
            threshold_type = threshold_types[attempt % len(threshold_types)]

            print(f"Попытка {attempt + 1}: blur_ksize={blur_ksize}, threshold_type={threshold_type}, block_size={block_size}")
            contours = detect_table(gray, blur_ksize, threshold_type, block_size)
            table_image = extract_table(image, contours)

            if table_image is not None:
                save_image(table_image, output_path)
                print(f"Таблица успешно извлечена и сохранена в {output_path}")
                return
        print("Не удалось найти таблицу на изображении после 10 попыток")
    except FileNotFoundError as e:
        print(e)

# Пример использования
tmp = ('Test4')
main(f'signatures/Test4_final1.jpg', f'signatures/{tmp}_final2.jpg')
