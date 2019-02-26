import numpy as np
    
def mean(image):
'''
This function takes an image and return the mean value of its pixels.
'''
    return image.mean()
        


def mean_center(image):
''' 
This function takes as an input an image and center it around 
the mean value of the pixels and returns the image.
'''
    #converting image to np.array
    img = np.array(image)
    mean_value = mean(img)
    
    return mean_value - img 

