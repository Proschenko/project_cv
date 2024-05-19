import cv2
import numpy as np


def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return edges


def find_lines(image):
    lines = cv2.HoughLinesP(image, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
    return lines


def compute_rotation_angle(lines):
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        angles.append(angle)
    median_angle = np.median(angles)
    return median_angle


def rotate_image(image, angle):
    # Получение размеров изображения
    h, w = image.shape[:2]

    # Определение центра изображения
    center = (w // 2, h // 2)

    # Вычисление матрицы поворота
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Выполнение поворота изображения
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR)

    # Заполнение черных областей
    rotated_image[rotated_image == 0] = 255  # Приравниваем черные пиксели к белым

    return rotated_image



def main(input_image_path, output_image_path):
    # Чтение изображения
    image = cv2.imread(input_image_path)

    # Предварительная обработка изображения
    processed_image = preprocess_image(image)

    # Поиск линий на изображении
    lines = find_lines(processed_image)

    # Вычисление угла поворота
    rotation_angle = compute_rotation_angle(lines)

    # Поворот изображения
    rotated_image = rotate_image(image, rotation_angle)

    # Сохранение результата
    cv2.imwrite(output_image_path, rotated_image)


if __name__ == "__main__":
    tmp = 'Test3'
    main(f'data_help/{tmp}.jpg', f'Dataset outter/{tmp}_out.jpg')
