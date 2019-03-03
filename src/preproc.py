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

def merge_images(img_data):
    '''
    This function taks as input folder containing the images.
    Merges all the images in the folder to one image and return
    the merged image.
    '''
    #copying the first image in the folder into a variable
    img1 = img_data[0]

    #looping through the rest of the images
    for j in range(1, len(img_data)):
        #copying the current image in a variable
        img2 = img_data[j]
        #merging the current image with the first image and overwriting the first image with the new merged image
        img1 = cv.addWeighted(img1, 0.5, img2, 0.5, 0)
    #reutrning the final merged image
    return img1

def normalize_img(grey_image):
    '''
    This function takes as an input a greyscale image, normalize the image and
    returns the normalized image.
    '''

    #finding the minimum pixel value in the image.
    min_val = np.amin(grey_image)
    #finding the maximum pixel value in the image
    max_val = np.amax(grey_image)
    #finding the range of the pixel values and storing in a variable.
    range_val = max_val - min_val

    #dividing each pixel value with the range to get the normalized image
    grey_image = grey_image/range_val
    return grey_image

def sobel_operator(img):
    '''
    This function takes as input an image applies sobel operator to it 
    and returns the image.
    '''
    return sobel(img)

def roberts_operator(img):
    '''
    This function takes as input an image applies roberts operator to it 
    and returns the image.
    '''
    return roberts(img)
