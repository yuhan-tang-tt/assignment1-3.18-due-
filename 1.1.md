1.1 Question 1: Area of a triangle

    import numpy as np

    def triangle_area_heron(a, b, c):

        # Create array of zeros same shape as input
        area = np.zeros_like(a, dtype=float)
    
        # Calculate s for all values
        s = ((a + b) + c)/2
    
        # Use numpy where to handle the triangle inequality
        valid = (a + b > c) & (b + c > a) & (a + c > b)
        area = np.where(valid, np.sqrt(s*(s - a)*(s - b)*(s - c)), np.nan)
    
        return area
