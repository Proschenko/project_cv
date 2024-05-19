from PIL import Image
import os

# Указываем пути к исходной и целевой папкам
source_folder = 'Dataset inner'
destination_folder = 'data_help'

# Создаем целевую папку, если она не существует
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Проходимся по всем файлам в исходной папке
for filename in os.listdir(source_folder):
    if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):  # Добавьте другие форматы, если нужно
        # Открываем изображение
        img_path = os.path.join(source_folder, filename)
        with Image.open(img_path) as img:
            # Поворачиваем изображение на 90 градусов по часовой стрелке
            rotated_img = img.rotate(-90, expand=True)

            # Сохраняем повернутое изображение в целевой папке
            save_path = os.path.join(destination_folder, filename)
            rotated_img.save(save_path)

print("Все изображения успешно повернуты и сохранены.")
