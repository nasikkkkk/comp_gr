import numpy as np
import math
import random
from PIL import Image

def load_triangles_and_vertices_from_obj(path: str) -> tuple[list[list[int]], list[list[float]]]:#Загружает индексы вершин для треугольников и координаты вершин из OBJ файла.

    faces = []
    vertices = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.startswith('v '):
                # tсли строка начинается с "v ", это значит, что это координаты вершины
                vertex_coords = list(map(float, line[2:].strip().split(' ')))
                vertices.append(vertex_coords)
            elif line.startswith('f '):
                # Если строка начинается с "f ", это значит, что это определение грани (треугольника)
                face_indices = line[2:].strip().split(' ')
                # Преобразовать в индексацию с нуля и сохранить только индекс вершин
                face_indices = [int(index.split('/')[0]) - 1 for index in face_indices]
                faces.append(face_indices)
    return faces, vertices


def barycentric_coordinates(x, y, x0, y0, x1, y1, x2, y2) -> tuple[float, float, float]:# вычисляем барицентрические координаты точки (x, y) внутри треугольника.
   
    det = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    if abs(det) < 1e-6:  #вырожд.треуг
        return -1, -1, -1 

    lambda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / det
    lambda1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / det
    lambda2 = 1.0 - lambda0 - lambda1
    return lambda0, lambda1, lambda2


def draw_triangle(image_array, x0, y0, x1, y1, x2, y2, color):# рисуем треугольник на заданном массиве изображения, используя барицентрические координаты.
    
    height, width, _ = image_array.shape

    # ограничивающий прямоугольник 
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


def create_image(H, W, filename, vertices, faces):#создание изображения

    
    image_array = np.full((H, W, 3), 255, dtype=np.uint8)  
    scale = min(H, W) * 5  
    offset_x = W // 2 
    offset_y = H // 2  

    for face in faces:
        v0 = vertices[face[0]]  # координаты вершин треугольника
        v1 = vertices[face[1]]
        v2 = vertices[face[2]]

        #  координаты вершин в координаты изображения
        x0 = v0[0] * scale + offset_x
        y0 = v0[1] * scale + offset_y
        x1 = v1[0] * scale + offset_x
        y1 = v1[1] * scale + offset_y
        x2 = v2[0] * scale + offset_x
        y2 = v2[1] * scale + offset_y

        color = [random.randint(0, 255) for _ in range(3)]  # случайный цвет для каждого треугольника
        draw_triangle(image_array, x0, y0, x1, y1, x2, y2, color)

    image = Image.fromarray(image_array, mode="RGB")  # создаем изображение из массива NumPy
    image.save(filename) 

if __name__ == '__main__':
    obj_file = "model_1.obj"  
    faces, vertices = load_triangles_and_vertices_from_obj(obj_file)
    create_image(1000, 1000, 'image.png', vertices, faces)
    print("Изображение сохранено как image.png")