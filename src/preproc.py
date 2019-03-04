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
    normalized_data = []
    for i in range(0,len(whole_data)):
        normalize_data_img = []
        for j in range(0, len(whole_data[i])):
            min_val = np.amin(whole_data[i][j])
            max_val = np.amax(whole_data[i][j])
            range_val = max_val - min_val
            grey_image = whole_data[i][j] - min_val/range_val
            normalize_data_img.append(grey_image)
            #deleting the previous image from the memory
            del grey_image
        normalized_data.append(normalize_data_img)
        del normalize_data_img
    return normalized_data

def reshape_image(img, x,y):
    return cv.resize(img,(x,y), interpolation = cv.INTER_CUBIC)
