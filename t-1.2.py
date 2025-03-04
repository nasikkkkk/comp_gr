import numpy as np
from PIL import Image


H = 600  
W = 800

image_data = np.full((H, W), 255, dtype=np.uint8)

# изображение из матрицы
image = Image.fromarray(image_data)
image.save('white_image.png')
