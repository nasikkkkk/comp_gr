import numpy as np
from PIL import Image

def v_fromFile(path: str):
    List = []
    
    with open(path, 'r', encoding='utf-8') as file:  
        for line in file:
            if line.startswith('v '):  # начало строки v
                line = line[2:].strip()  
                List.append(list(map(float, line.split(' '))))
    return List

def create_image(H, W, filename, List):
    image_array = np.zeros((H, W, 3), dtype=np.uint8)
    
    for ls in List:
        x = int(ls[0] * 5000 + 500)
        y = int(ls[1] * 5000 + 250)

    image = Image.fromarray(image_array, mode="RGB")
    image.save(filename)

List = v_fromFile("model_1.obj")
create_image(1000, 1000, 'image.png', List)
print(List)
