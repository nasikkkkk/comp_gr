def barycentric_coordinates(x, y, x0, y0, x1, y1, x2, y2):
  

    denominator = (x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2)

    if denominator == 0:
        
        return 0, 0, 0

    lambda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / denominator
    lambda1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / denominator
    lambda2 = 1.0 - lambda0 - lambda1

    return lambda0, lambda1, lambda2