import numpy as np
import math
import random
from PIL import Image

# Размеры изображения
w = 1000
h = 1000

# Z-буфер (буфер глубины)
Z_BUFFER = np.zeros((w, h), dtype=np.float32)
Z_BUFFER[...] = np.inf  # Инициализация Z-буфера (все значения - бесконечность)

def f_fromFile(path:str):
    
    List = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            # Проверяем, является ли строка описанием грани (начинается с "f ")
            if line[0] == 'f' and line[1] == ' ':
                line = line[2:-1]  
                # Извлекаем индексы вершин для грани, обрабатывая возможные
                List.append(list(map(int, [line.split(' ')[0].split('/')[0], line.split(' ')[1].split('/')[0], line.split(' ')[2].split('/')[0]])))
    return List

def v_fromFile(path:str):
    
    List = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            # Проверяем, является ли строка описанием вершины (начинается с "v ")
            if line[0] == 'v' and line[1] == ' ':
                line = line[2:-1]  # Удаляем префикс "v " и символ новой строки
                # Извлекаем координаты вершины
                List.append(list(map(float, line.split(' '))))
    return List

def bar_coord(x, y, x0, y0, x1, y1, x2, y2)->tuple:
    
    # Вычисление барицентрических координат по формуле
    lambda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) /  ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda2 = 1.0 - lambda0 - lambda1  # lambda0 + lambda1 + lambda2 = 1
    return (lambda0, lambda1, lambda2)

def draw_triangle(mat, x0, y0, z0, x1, y1, z1, x2, y2, z2, color):
    
    # Предварительные вычисления констант для эффективности
    scaled_time = 200  # Коэффициент масштабирования для перспективной проекции
    sdvig = 500  # Сдвиг (смещение) для центрирования изображения

    # Перспективная проекция: масштабировать x, y на 1/z и выполнить смещение
    proj_x0, proj_y0 = scaled_time * x0 / z0 + sdvig, scaled_time * y0 / z0 + sdvig
    proj_x1, proj_y1 = scaled_time * x1 / z1 + sdvig, scaled_time * y1 / z1 + sdvig
    proj_x2, proj_y2 = scaled_time * x2 / z2 + sdvig, scaled_time * y2 / z2 + sdvig

    # Вычисление ограничивающей рамки треугольника
    xmin = max(0, math.floor(min(proj_x0, proj_x1, proj_x2)))  
    ymin = max(0, math.floor(min(proj_y0, proj_y1, proj_y2)))  
    xmax = min(w, math.ceil(max(proj_x0, proj_x1, proj_x2)))  
    ymax = min(h, math.ceil(max(proj_y0, proj_y1, proj_y2))) 


   
    for y in range(ymin, ymax):  # Перебор строк изображения
        for x in range(xmin, xmax):  # Перебор столбцов изображения
            # Вычисление барицентрических координат *1 раз для каждого пикселя*
            coords = bar_coord(x, y, proj_x0, proj_y0, proj_x1, proj_y1, proj_x2, proj_y2)

            # Проверка, находится ли точка внутри треугольника
            if all(coord >= 0 for coord in coords):  # Если все барицентрические координаты неотрицательны
                # Интерполяция Z-значения *один раз для каждого пикселя*
                z_val = coords[0] * z0 + coords[1] * z1 + coords[2] * z2  # Интерполируем Z-координату

                # Z-буферизация (проверка глубины)
                if z_val < Z_BUFFER[y, x]:  # Если текущая глубина меньше, чем в буфере
                    Z_BUFFER[y, x] = z_val  # Обновляем Z-буфер новым значением глубины
                    mat[y, x] = color  # Устанавливаем цвет пикселя в буфере изображения

def normal(x0, y0, z0, x1, y1, z1, x2, y2, z2):
   
    # Вычисление нормали с использованием векторного произведения
    n = np.cross(np.array([x1 - x2, y1 - y2, z1 - z2]), np.array([x1 - x0, y1 - y0, z1 - z0]))
    return n

def compute_lighting(normal_vector):
    
    light_direction = np.array([0, 0, 1])  # Направление источника света
    norm_length = np.linalg.norm(normal_vector)  # Вычисление длины вектора нормали

    # Обработка случая, когда вектор нормали имеет нулевую длину (вырожденный треугольник)
    if norm_length == 0:
        return 0.0

    # Вычисление косинуса угла между вектором нормали и направлением света
    cosine_angle = np.dot(normal_vector, light_direction) / norm_length
    return cosine_angle

def rotate_and_translate(vertices, alpha, beta, gamma, translation=(0, 0.03, 0.1)):
  
    # Матрицы вращения
    Rx = np.array([[1, 0, 0], [0, math.cos(alpha), math.sin(alpha)], [0, -math.sin(alpha), math.cos(alpha)]])
    Ry = np.array([[math.cos(beta), 0, math.sin(beta)], [0, 1, 0], [-math.sin(beta), 0, math.cos(beta)]])
    Rz = np.array([[math.cos(gamma), math.sin(gamma), 0], [-math.sin(gamma), math.cos(gamma), 0], [0, 0, 1]])

    # Комбинированная матрица вращения
    rotation_matrix = Rz @ Ry @ Rx
    translation_vector = np.array(translation)  # Вектор переноса
    transformed_vertices = [rotation_matrix @ vertex + translation_vector for vertex in vertices]  # Поворачиваем и переносим
    return transformed_vertices

def generate_image(height, width, output_filename, vertices, faces):
    
    image_buffer = np.full((height, width, 3), 255, dtype=np.uint8)  # Инициализируем буфер изображения белым цветом
    
    # Определяем углы поворота (в радианах)
    alpha_angle = math.radians(160)
    beta_angle = math.radians(-0.5)
    gamma_angle = math.radians(0)

    # Поворачиваем и переносим вершины
    rotated_verts = rotate_and_translate(vertices, alpha_angle, beta_angle, gamma_angle)

    # Итерируемся по каждой грани
    for face_index in range(len(faces)):
        # Получаем индексы вершин для текущей грани
        v1_idx, v2_idx, v3_idx = faces[face_index]

        # Получаем координаты вершин
        v1 = rotated_verts[v1_idx - 1]
        v2 = rotated_verts[v2_idx - 1]
        v3 = rotated_verts[v3_idx - 1]

        # Вычисляем вектор нормали треугольника
        triangle_normal = normal(v1[0], v1[1], v1[2], v2[0], v2[1], v2[2], v3[0], v3[1], v3[2])

        # Вычисляем интенсивность освещения
        cosine_angle = compute_lighting(triangle_normal)

        # Выполняем отсечение невидимых граней (backface culling)
        if cosine_angle < 0:
            shade = -cosine_angle * 255  # Вычисляем оттенок серого
            draw_triangle(image_buffer, v1[0], v1[1], v1[2], v2[0], v2[1], v2[2], v3[0], v3[1], v3[2], shade)  # Рисуем треугольник

    # Создаем изображение из буфера и сохраняем его в файл
    final_image = Image.fromarray(image_buffer, mode="RGB")
    final_image.save(output_filename)

# Загружаем данные о гранях и вершинах из OBJ файла
f = f_fromFile("model_1.obj")
v = v_fromFile("model_1.obj")

# Генерируем изображение
generate_image(w, h, 'image_z_buffer.png', v, f)