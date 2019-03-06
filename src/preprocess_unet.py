
import preproc as pre
import numpy as np



def normalize_img_unet(data):
    '''
    This function takes as input the data as list of images and normalizes it.
    '''
    normalize_unet_data = []
    for j in range(0,len(data)):
        min_val = np.amin(data[j]) # finding the maximum value of the image
        max_val = np.amax(data[j]) # finding the minimum value of the image
        range_val = max_val - min_val # finding the range of pixel values in the image
        norm_img = data[j] - min_val/range_val # normalizing the image
        normalize_unet_data.append(norm_img) # appending the normalized image to the list
        del norm_img # deleting the current image from the memory
    return normalize_unet_data

def create_unet_data(img_list):
    train_nparray=np.ndarray(shape=(len(image_list), 256, 256, 1),
                     dtype=np.float32)
    for i in range(0,len(image_list)):
        train_nparray[i]=image_list[i]

    return train_nparray

def shape_img_data(data):
    '''
    This function takes as data as list of list and fit the data to be passed to the model.
    Returns the modified data
    '''
    merged_data = pre.merge_images(data) # calling the function to merge 100 images into 1
    del whole_data # deleting the original data from the memory

    normalize_data = normalize_img_unet(merged_data) #  calling the function to normalize the merged images
    del merged_data #deleting the merged images from the memory

    reshaped_data = pre.reshape_image(normalize_data,256,256) # calling the function to reshape the normalized data
    del normalize_data # deleting the normalized data from the memory

    axis_img = pre.add_axis(reshaped_data) # calling the function to add axis to the images
    train_unet = create_unet_data(axis_img) 
    del axis_img # deleting the previous data from the memory
    return train_unet 

def shape_mask_data(mask_list):
    '''
    This function takes as input the list of images i.e masks and process these images to pass it to the model.
    Returns the modified images as a list.
    '''
    reshaped_mask = pre.reshape_image(mask_list,256,256) # calling the function to reshape the image
    del mask_list  # deleting the original images from the memory

    mask_axis = pre.add_axis(reshaped_mask) # calling the function to add axis to the images
    del reshaped_mask # deleting the previous images from the memory

    train_mask_unet = create_unet_data(mask_axis)
    return train_mask_unet
