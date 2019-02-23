import numpy as np
    
def mean(image):
    return image.mean()
        


def mean_center(image):

    #converting image to np.array
    img = np.array(image)
    
    mean_img = mean(img)


    return mean_img - img 

