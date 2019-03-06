import numpy as np

def mean(image):
'''
This function takes an image and return the mean value of its pixels.
'''
    return image.mean()



def mean_center(whole_data):
    '''
    This function takes as input whole data which is the list of list
    and perform mean centering on all the frames and return the list of list.
    '''
    mean_centered_data = []
    mean_img_data = []
    for i in range(0,len(whole_data)):
        for j in range(0,len(whole_data[i])):
            #converting the image to numpy array
            img = np.array(whole_data[i][j])
            #passing the img to the function to calculate the mean
            mean_value = mean(img)
            #subtractin each pixel of the image from mean
            img = mean_value - img
            #appending the image to the list
            mean_img_data.append(img)
            #removing the previously stored image from the memory
            del img
        #appending the list of frames into a new list
        mean_centered_data.append(mean_img_data)
        #deleting the previous list of images from the memory
        del mean_img_data

    return mean_centered_data

def merge_images(whole_data):
    '''
    This function takes as input the list of list of the whole data, merge all the
    frames into one and returns a list containing one frame per file.
    '''
    merged_data = []

    #looping through the
    for i in range(0, len(whole_data)):
        #storing the first frame for the ith file
        img1 = whole_data[i][0]

        #looping through all the frames of ith file
        for j in range(1, len(whole_data[i])):
            img2 = whole_data[i][j]
            #merging img2 and img1 into one and overwriting img1
            img1 = cv.addWeighted(img1, 0.5, img2, 0.5, 0)
        #after merging all the images appending the final image in the list
        merged_data.append(img1)
    #returning the list
    return merged_data

def normalize_img(whole_data):
    '''
    This function takes as input the list of list containing the data. Perform normalization 
    on each image in the data and returns a list of list.
    '''
    normalized_data = []
    for i in range(0,len(whole_data)): 
        normalize_data_img = []
        for j in range(0, len(whole_data[i])):
            min_val = np.amin(whole_data[i][j]) #finding the minmimum value of the image
            max_val = np.amax(whole_data[i][j]) # finding the maximum value of the image
            range_val = max_val - min_val #getting the range
            grey_image = whole_data[i][j] - min_val/range_val #normalizing the image by subtracting minimum value from each pixel and then dividing it by the range
            normalize_data_img.append(grey_image) #appending the image to the list
            del grey_image #deleting current image from memory
        normalized_data.append(normalize_data_img) #appending the list of images to a list
        del normalize_data_img #deleting the list of image from the memory
    return normalized_data

def reshape_image(img, x,y):
    '''
    This function takes as an input an image and x and y two integer values. Resizes the image to the image of size x*y
    and returns the new resized image.
    '''
    return cv.resize(img,(x,y), interpolation = cv.INTER_CUBIC)
