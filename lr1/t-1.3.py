import numpy as np
from PIL import Image
H = 600  
W = 800

image_data = np.zeros((H, W, 3), dtype=np.uint8)
image_data[:] = [255, 0, 0]  


image = Image.fromarray(image_data)

image.save('red_image.png')