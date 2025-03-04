import numpy as np
import math
from PIL import Image

def dotter_line(image, x0, y0, x1, y1, color):
    
    step = 1.0 
    for t in np.arange(0, 1, step):
        x = round((1.0 - t) * x0 + t * x1)
        y = round((1.0 - t) * y0 + t * y1)
        image[y, x] = color

def dotter_line2(image, x0, y0, x1, y1, color):
    count = math.sqrt((x0 - x1)**2 + (y0 - y1)**2)
    step = 1.0 / count 
    for t in np.arange(0, 1, step):
        x = round((1.0 - t) * x0 + t * x1)
        y = round((1.0 - t) * y0 + t * y1)
        image[y, x] = color

def x_loop_line(image, x0, y0, x1, y1, color):#если нач. точка правее
    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    for x in range(x0, x1):  
        t = (x - x0) / (x1 - x0) 
        y = round((1.0 - t) * y0 + t * y1)
        image[y, x] = color

def x_loop_line2(image, x0, y0, x1, y1, color):#если по Х больше то меняем местами 
    xchange = False
    if abs(x0 - x1) < abs(y0 - y1):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True

    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    for x in range(x0, x1):  
        t = (x - x0) / (x1 - x0) 
        y = round((1.0 - t) * y0 + t * y1)
        if xchange:
                image[x, y] = color
        else:
                image[y, x] = color
def x_loop_line_v2(image, x0, y0, x1, y1, color):
    xchange = False
    if (abs(x0 - x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True
    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    y = y0
    dy = abs(y1 -y0)/(x1 -x0)
    derror = 0.0
    y_update = 1 if y1 > y0 else -1
    for x in range (x0, x1):
        if (xchange):
            image[x, y] = color
        else:
            image[y, x] = color
        derror += dy
        if (derror > 0.5):
            derror-= 1.0
            y += y_update 
def x_loop_line_v2_2(image, x0, y0, x1, y1, color):#Алгоритм Брезенхема
    xchange = False
    if (abs(x0 - x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True
    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    y = y0
    dy = 2*abs(y1 -y0)
    derror = 0.0
    y_update = 1 if y1 > y0 else -1
    for x in range (x0, x1):
        if (xchange):
           image[x, y] = color
        else:
            image[y, x] = color
        derror += dy
        if derror > 2*(x1 - x0)*0.5:
            derror -= 2.0*(x1 - x0)*1.0
            y += y_update 
def cremage(H, W, filename):
    image_array = np.zeros((H, W, 3), dtype=np.uint8)
    
    x0 = 100
    y0 = 100
    for t in range(13):
        x1 = int(100 + 95 * math.cos(t * 2 * math.pi / 13))
        y1 = int(100 + 95 * math.sin(t * 2 * math.pi / 13))
       # dotter_line(image, x0, y0, x1, y1, [255, 0, 0])
        #dotter_line2(image, x0, y0, x1, y1, [255, 0, 0])
        #x_loop_line(image_array, x0, y0, x1, y1, [255, 0, 0]) 
       # x_loop_line2(image_array, x0, y0, x1, y1, [255, 0, 0]) 
        #x_loop_line_v2(image, x0, y0, x1, y1, [255, 0, 0])
        x_loop_line_v2_2 (image_array, x0, y0, x1, y1, [255, 0, 0])

    image = Image.fromarray(image_array, mode="RGB")
    image.save(filename)

cremage(200, 200, 'image.png')
