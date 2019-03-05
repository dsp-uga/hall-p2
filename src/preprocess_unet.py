
import preproc as pre
import numpy as np



def normalize_img_unet(data):
    normalize_unet_data = []
    for j in range(0,len(data)):
        min_val = np.amin(data[j])
        max_val = np.amax(data[j])
        range_val = max_val - min_val
        norm_img = data[j] - min_val/range_val
        normalize_unet_data.append(norm_img)
        del norm_img
    return normalize_unet_data

def create_unet_data(img_list):
    train_nparray=np.ndarray(shape=(len(image_list), 256, 256, 1),
                     dtype=np.float32)
    for i in range(0,len(image_list)):
        train_nparray[i]=image_list[i]

    return train_nparray

def shape_img_data(data):
    merged_data = pre.merge_images(data)
    del whole_data

    normalize_data = normalize_img_unet(merged_data)
    del merged_data

    reshaped_data = pre.reshape_image(normalize_data,256,256)
    del normalize_data

    axis_img = pre.add_axis(reshaped_data)
    train_unet = create_unet_data(axis_img)
    del axis_img
    return train_unet

def shape_mask_data(mask_list):
    reshaped_mask = reshape_image(mask_list)
    del mask_list

    mask_axis = add_axis(reshaped_mask)
    del reshaped_mask

    train_mask_unet = create_unet_data(mask_axis)
    return train_mask_unet
