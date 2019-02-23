import numpy as np
    
def mean(image):
        
    # converting the image to np.array
    img = np.array(image)

    #initializing sum with 0
    sum = 0 

    #loop to run through all the pixel values and calculate their sum
    for i in np.nditer(img):
        sum = sum + i


    #finding the mean of the image
    mean = sum/img.size
    return mean
