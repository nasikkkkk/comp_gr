import numpy as np
from PIL import Image
H = 600  
W = 800  


image_data = np.zeros((H, W, 3), dtype=np.uint8)


for y in range(H):
    for x in range(W):
        value = (x + y) %160
        image_data[y, x] = [value, value, value] 


image = Image.fromarray(image_data)


image.save('gradient_image.png')

