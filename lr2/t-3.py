import numpy as np
import math
import random
from PIL import Image

def load_triangles_and_vertices_from_obj(path: str) -> tuple[list[list[int]], list[list[float]]]:
    
    faces = []
    vertices = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.startswith('v '):
                #  координаты вершины
                vertex_coords = list(map(float, line[2:].strip().split(' ')))
                vertices.append(vertex_coords)
            elif line.startswith('f '):
                # строка начинается с "f ",  это определение грани (треугольника)
                face_indices = line[2:].strip().split(' ')
                # преобразовать в индексацию с нуля и сохранить только индекс вершин
                face_indices = [int(index.split('/')[0]) - 1 for index in face_indices]
                faces.append(face_indices)
    return faces, vertices


def barycentric_coordinates(x, y, x0, y0, x1, y1, x2, y2) -> tuple[float, float, float]:
  
    det = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    if abs(det) < 1e-6:  
        return -1, -1, -1 

    lambda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / det
    lambda1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / det
    lambda2 = 1.0 - lambda0 - lambda1
    return lambda0, lambda1, lambda2


def draw_triangle(image_array, x0, y0, x1, y1, x2, y2, color):
    
    height, width, _ = image_array.shape

    # ограничивающий прямоугольник (bounding box)
    xmin = math.floor(min(x0, x1, x2))
    ymin = math.floor(min(y0, y1, y2))
    xmax = math.ceil(max(x0, x1, x2))
    ymax = math.ceil(max(y0, y1, y2))

    # обрезка по границам изображения
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(width, xmax)
    ymax = min(height, ymax)

    for x in range(xmin, xmax):
        for y in range(ymin, ymax):
            # вычисление барицентрических координат для текущего пикселя
            lambda0, lambda1, lambda2 = barycentric_coordinates(x, y, x0, y0, x1, y1, x2, y2)
            if lambda0 >= 0 and lambda1 >= 0 and lambda2 >= 0:
                # если все координаты неотрицательные, то пиксель находится внутри треугольника
                image_array[y, x] = color 


def calculate_normal(v0, v1, v2):
    
    v0 = np.array(v0)
    v1 = np.array(v1)
    v2 = np.array(v2)
    normal = np.cross(v1 - v0, v2 - v0)  # векторное произведение
    return normal


def calculate_cosine_angle(normal, light_direction=[0, 0, 1]):#вычисляет косинус угла между нормалью и направлением света.Используется для отсечения нелицевых граней и для расчета освещения.
     # преобразуем входные данные в NumPy массивы для удобной работы с векторами
    normal = np.array(normal)
    light_direction = np.array(light_direction)

    # нормализация векторов: приводим векторы к единичной длине.
    # это необходимо, чтобы скалярное произведение правильно отражало угол.
    # делим каждый компонент вектора на его длину.
    normal = normal / np.linalg.norm(normal)
    light_direction = light_direction / np.linalg.norm(light_direction)

    # вычисляем скалярное произведение нормализованных векторов.
    # скалярное произведение равно косинусу угла между векторами (т.к. векторы нормализованы).
    cosine = np.dot(normal, light_direction)

    return cosine


def create_image(H, W, filename, vertices, faces):#Создает изображение с отрисовкой трехмерной модели.Применяет отсечение нелицевых граней и базовое освещение.
    
    image_array = np.full((H, W, 3), 0, dtype=np.uint8)  # Инициализация черным цветом
    scale = min(H, W)*5
    offset_x = W // 2  
    offset_y = H *0.05 

    for face in faces:
        # Получаем координаты вершин
        v0 = vertices[face[0]]
        v1 = vertices[face[1]]
        v2 = vertices[face[2]]

        # Вычисляем нормаль
        normal = calculate_normal(v0, v1, v2)

        # Вычисляем косинус угла между нормалью и направлением света
        cosine = calculate_cosine_angle(normal)

        # Отсечение нелицевых граней: рисуем, только если грань лицевая (cosine < 0)
        if cosine < 0:
            # вычисляем цвет в зависимости от косинуса угла (базовое освещение)
            intensity = -cosine  # Отрицательное значение косинуса, так как отсекаем задние грани
            gray_value = int(150 * intensity) # Light Gray
            color = [gray_value, gray_value, gray_value]  # базовое освещение: серый цвет

            
            x0 = v0[0] * scale + offset_x
            y0 = v0[1] * scale + offset_y
            x1 = v1[0] * scale + offset_x
            y1 = v1[1] * scale + offset_y
            x2 = v2[0] * scale + offset_x
            y2 = v2[1] * scale + offset_y

            # рисуем треугольник
            draw_triangle(image_array, x0, y0, x1, y1, x2, y2, color)

    image = Image.fromarray(image_array, mode="RGB")  # создаем изображение из массива NumPy
    image.save(filename)  # сохраняем изображение
    print("image.png")


if __name__ == '__main__':
    obj_file = "model_1.obj"  
    faces, vertices = load_triangles_and_vertices_from_obj(obj_file)
    create_image(1000, 1000, 'image.png', vertices, faces)
    print("image.png")