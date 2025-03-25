import numpy as np
import math
import random
from PIL import Image


def load_triangles_and_vertices_from_obj(path: str) -> tuple[list[list[int]], list[list[float]]]:
    """
    Загружает индексы вершин для треугольников (грани) и координаты вершин из OBJ файла.

    """
    faces = []#представляет собой треугльник и и содержит индексы вершин для этой грани.  Индексы начинаются с 0.
    vertices = []#список списков, где каждый внутренний список представляет вершину и содержит x, y и z координаты этой вершины.
    try:
        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                if line.startswith('v '):
                    # Если строка начинается с "v ", это координаты вершины
                    vertex_coords = list(map(float, line[2:].strip().split(' ')))
                    vertices.append(vertex_coords)
                elif line.startswith('f '):
                    # Если строка начинается с "f ", это определение грани (треугольника)
                    face_indices = line[2:].strip().split(' ')
                    # Преобразовать в индексацию с нуля и сохранить только индекс вершин
                    # OBJ файлы используют 1-индексацию, поэтому вычитаем 1
                    face_indices = [int(index.split('/')[0]) - 1 for index in face_indices]
                    faces.append(face_indices)
    except FileNotFoundError:
        print(f"Ошибка: Файл '{path}' не найден.")
        return [], []
    except Exception as e:
        print(f"Ошибка при чтении файла '{path}': {e}")
        return [], []
    return faces, vertices



def barycentric_coordinates(x, y, x0, y0, x1, y1, x2, y2) -> tuple[float, float, float]:#Вычисляет барицентрические координаты точки (x, y) внутри треугольника.
    
       
    
    det = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    if abs(det) < 1e-6:  # вырожд.треуг

        return -1, -1, -1 #невалидный треугольник

    w0 = ((x1 - x) * (y2 - y) - (x2 - x) * (y1 - y)) / det
    w1 = ((x2 - x) * (y0 - y) - (x0 - x) * (y2 - y)) / det
    w2 = 1 - w0 - w1

    return w0, w1, w2


def calculate_normal(v0, v1, v2):# 11. вычисление нормали к поверхности треугольника.Вычисляет нормаль к треугольнику по координатам его вершин.
    v0 = np.array(v0)
    v1 = np.array(v1)
    v2 = np.array(v2)
    normal = np.cross(v1 - v0, v2 - v0)  # векторное произведение
    return normal


def calculate_cosine_angle(normal, light_direction=[0, 0, 1]):
    
    #12. отсечение нелицевых граней / Базовое освещение/Вычисляет косинус угла между нормалью и направлением света./Используется для отсечения нелицевых граней и для расчета освещения.

    normal = np.array(normal)#вектор нормали
    light_direction = np.array(light_direction)#направление света

    # Нормализуем векторы, чтобы избежать ошибок при малых значениях
    if np.linalg.norm(normal) != 0:
        normal = normal / np.linalg.norm(normal)

    if np.linalg.norm(light_direction) != 0:
        light_direction = light_direction / np.linalg.norm(light_direction)

    cosine = np.dot(normal, light_direction)
    return cosine



def draw_triangle(image_array, x0, y0, z0, x1, y1, z1, x2, y2, z2, color, z_buffer):
   
    #отрисовывает треугольник на заданном массиве изображения, используя барицентрические координаты и Z-буфер.

   
    height, width, _ = image_array.shape

    # ограничивающий прямоугольник 
    xmin = int(max(0, min(x0, x1, x2)))
    ymin = int(max(0, min(y0, y1, y2)))
    xmax = int(min(width - 1, max(x0, x1, x2)))  # Corrected xmax
    ymax = int(min(height - 1, max(y0, y1, y2))) # Corrected ymax

    # Перебор пикселей внутри ограничивающего прямоугольника
    for y in range(ymin, ymax + 1):
        for x in range(xmin, xmax + 1): 
            # вычисляем барицентрические координаты для текущего пикселя
            w0, w1, w2 = barycentric_coordinates(x, y, x0, y0, x1, y1, x2, y2)

            # если пиксель находится внутри треугольника (все координаты >= 0)
            if w0 >= 0 and w1 >= 0 and w2 >= 0:
                # вычисляем z-координату для текущего пикселя
                z = w0 * z0 + w1 * z1 + w2 * z2

                # если z-координата меньше, чем текущее значение в Z-буфере,
                # то пиксель ближе , и мы его рисуем
                if z < z_buffer[y, x]:
                    # Обновляем значение в Z-буфере
                    z_buffer[y, x] = z

                    # отрисовываем пиксель
                    image_array[y, x] = color


def create_image(H, W, filename, vertices, faces):
   
    #10. Отрисовка полигонов трёхмерной модели12. Отсечение нелицевых граней13. Базовое освещение14. z-буфер/Создает изображение с отрисовкой трехмерной модели, используя Z-буфер,
    

    image_array = np.full((H, W, 3), 0, dtype=np.uint8)  # Инициализация черным цветом
    z_buffer = np.full((H, W), np.inf)  # инициализация Z-буфера (все точки "далеко")
    scale = min(H, W)*5  
    offset_x = W // 2 
    offset_y = H *0.05 
   

    light_direction = [0, 0, 1] # Определяем направление света

    for face in faces:
        # Получаем индексы вершин для текущего треугольника
        v0_index, v1_index, v2_index = face
        # Получаем координаты вершин из списка вершин
        try:
            v0 = vertices[v0_index]
            v1 = vertices[v1_index]
            v2 = vertices[v2_index]
        except IndexError:
            print(f"Ошибка: Неверный индекс вершины в грани: {face}")
            continue

        # 11. Вычисляем нормаль к поверхности треугольника
        normal = calculate_normal(v0, v1, v2)

        # 12. Отсечение нелицевых граней и 13. Базовое освещение
        cosine = calculate_cosine_angle(normal, light_direction)

        # Отсечение нелицевых граней: рисуем, только если грань "лицевая"
        if cosine < 0:
            # 13. Базовое освещение: Вычисляем цвет в зависимости от угла между нормалью и светом
            intensity = -cosine #Интенсивность (от 0 до 1)
            color = [int(255 * intensity), int(255*intensity), int(255*intensity)]  # Серый цвет, зависящий от интенсивности

            # Преобразуем мировые координаты в координаты изображения
            x0 = v0[0] * scale + offset_x
            y0 = v0[1] * scale 
            z0 = v0[2] * scale  #Масштабируем Z
            x1 = v1[0] * scale + offset_x
            y1 = v1[1] * scale 
            z1 = v1[2] * scale #Масштабируем Z
            x2 = v2[0] * scale + offset_x
            y2 = v2[1] * scale 
            z2 = v2[2] * scale #Масштабируем Z

            # Рисуем треугольник
            draw_triangle(image_array, x0, y0, z0, x1, y1, z1, x2, y2, z2, color, z_buffer)

    # Сохраняем изображение
    image = Image.fromarray(image_array, mode="RGB")
    image.save(filename)
    print(f"Изображение сохранено")


if __name__ == '__main__':
    obj_file = "model_1.obj"  # Замените на имя вашего OBJ файла
    H, W = 600, 800         # Размеры изображения

    # Загружаем вершины и грани из OBJ файла
    faces, vertices = load_triangles_and_vertices_from_obj(obj_file)

    # Проверяем, загрузились ли данные
    if not vertices or not faces:
        print("Ошибка: Не удалось загрузить данные из OBJ файла")
    else:
        # создаем изображение с Z-буфером
        create_image(H, W, "bunny.png", vertices, faces)