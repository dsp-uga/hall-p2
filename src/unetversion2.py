
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras import backend as K
from skimage import io
import numpy as np
import tensorflow as tf
from unet_model import unet
import cv2 as cv
import os as os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



def get_train_names():
    '''
    Reads train.txt file from dataset folder and creates a list with all lines in train.txt file.
    The lines in train.txt contain ids for each individual folder containing respective frames.
    '''
    f = open("dataset/train.txt", "r")
    train_names=f.read().splitlines()
    print(len(train_names))
    return train_names

def get_test_names(): 
    '''
    Reads test.txt file from dataset folder and creates a list with all lines in train.txt file.
    The lines in train.txt contain ids for each individual folder containing respective frames.
    '''
    f = open("dataset/test.txt", "r")
    test_names=f.read().splitlines()
    print(len(test_names))
    return test_names


def data_video_dictionary(train_names): 
    
    """
    creates a dictionary of dictionary with following structure
    dict_videos --> key = train filename , value = dict_image --> key = frame no. of images, value = image of celia
    ie first dictonary size is 211 and every dictionary inside 211 dictionaries have size 100. 
    And dictionaries with size 100 contain the images by frames. 
    """
    
    length=0 # to check the length of total dataset
    dict_videos= {}
    for name in train_names: 
        dict_images={}    
        for i in range(len(os.listdir("dataset/data/"+name))): 
            dict_images[str(i)]= cv.imread("dataset/data/"+name+"/frame00"+"{:02d}".format(i)+".png",cv.IMREAD_GRAYSCALE)
            
        length=length+len(dict_images)
        dict_videos[name]=dict_images
    #what I am doing here is error checking, can do this professionally also..do it after experiments..
    print("Length of total dataset, should be 21100 :"+ str(length))    
    return dict_videos



def get_optical_dict(train_names,path="train_optical_flow"): 
    dict_optical={}
	"""
	result of optical is taken as an output from a different file. 
	"""
    for name in train_names: 
        dict_optical[name]= cv.imread(path+"/"+name+".png",cv.IMREAD_GRAYSCALE)        
    return dict_optical    



def variance_on_images(dict_videos,train_names,hop=1,scale=1): 
	"""
	calculate variance of 100 images and creates a dictionary of them with respective train or test lists. Note that train names are used 
	, but the function is general. 
	"""
    dict_videos_mean={}
    dict_videos_variance={}
    for name in train_names:
        dict_image= dict_videos[name]
        arr=np.zeros(dict_image[str(0)].shape,np.float)
        N_1=len(list(range(0,len(dict_image),hop)))
        N_2=len(list(range(0,len(dict_image))))
        #calculating mean of an image. 
        for i in range(0,len(dict_image),hop): 
            image=dict_image[str(i)]
            imarr=np.array(image,dtype=np.float)
            arr=arr+imarr/N_1
        image_mean=np.array(np.round(arr),dtype=np.uint8)
        dict_videos_mean[name]=image_mean #for later use, can send as results.
        varr=np.zeros(dict_image[str(0)].shape,np.float)
        for i in range(0,len(dict_image),hop): 
            #using arr because its an image_mean in float type
            image=dict_image[str(i)]
            imarr=np.array(image,dtype=np.float)
            varr=varr+ (scale*(imarr-arr)**2)/N_1
        variance_image =np.array(np.round(varr),dtype=np.uint8)
        dict_videos_variance[name]=variance_image
        #calculating square of X-mean
        #result action.     
    return dict_videos_mean,dict_videos_variance
#hop of 15 is used because Ciliary motion does 5-6 oscillations in 1 frame..so, 15*6 will capture those oscillations


def prepare_variance_data(dict_videos,train_names): 
    """
    prepares training data for processing ..
    steps :- 
    0)get wholedata too..
    1)Merge 100 frames and normalize the final images in all 211 
    2)take list of images , and convert them to 256,256.. ie resize them.
    3) stack images into (211,256,256,1)
    """
    dict_videos_mean,dict_videos_variance=variance_on_images(dict_videos,train_names,hop=4,scale=2)
    merged_images=[] #merged and normalized images..list
    optical_dict= get_optical_dict(train_names) if(len(train_names)>150) else get_optical_dict(test_names,"test_optical_flow")
       
    
    
    for name in train_names:
        dict_image= dict_videos[name]
        optical_image=optical_dict[name]
        img=merge_images(list(dict_image.values()))
        img_var = cv.addWeighted(dict_videos_variance[name], 0.5, optical_image, 0.5, 0) 
        img_optical=cv.addWeighted(img_var,0.3,img,0.7,0)
        new_img=normalize_img(img_optical)
        resized_img= reshape_image(new_img,256,256)
        axis_img=add_axis(resized_img)
        merged_images.append(axis_img)
    
    train_nparray=create_train_nparray(merged_images)

    return train_nparray




def create_train_nparray(image_list): 
    """creates nparray of (211,256,256,1)and saved it, out of a list of images, with expected dimensions of (256,256,1)"""
    train_nparray=np.ndarray(shape=(len(image_list), 256, 256, 1),
                     dtype=np.float32)
    for i in range(0,len(image_list)):
        train_nparray[i]=image_list[i]
    #np.save("train.npy",train_nparray)
    return train_nparray




def load_train(name="train.npy"): 
    train_nparray=np.load(name)
    return train_nparray



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
    grey_image = (grey_image - min_val)/range_val
    return grey_image



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



def add_axis(img):
    """use it for adding one axis ie the 1 dimensio in the end.."""
    return img[...,np.newaxis]


def reshape_image(img, x,y):
    return cv.resize(img,(x,y), interpolation = cv.INTER_CUBIC)



def prepare_train_data(dict_videos,train_names): 
    """
    prepares training data for processing ..
    steps :- 
    0)get wholedata too..
    1)Merge 100 frames and normalize the final images in all 211 
    2)take list of images , and convert them to 256,256.. ie resize them.
    3) stack images into (211,256,256,1)
    4) 
    """
    merged_images=[] #merged and normalized images..list
    for name in train_names:
        dict_image= dict_videos[name]
        img=merge_images(list(dict_image.values()))
        new_img=normalize_img(img)
        resized_img= reshape_image(new_img,256,256)
        axis_img=add_axis(resized_img)
        merged_images.append(axis_img)
    
    train_nparray=create_train_nparray(merged_images)

    return train_nparray



def preprocess_masks(): 
   """
   reads masks from the file, 
   converts it into nparray of dimensions (211,256,256,1) and returns it
   
   """
   mask_list=[]
   for names in train_names: 
       mask = cv.imread('dataset/masks/'+names+'.png',0)
       resized_img= reshape_image(mask,256,256)
       axis_img=add_axis(resized_img)
       mask_list.append(axis_img)     
   masks=create_train_nparray(mask_list)
   return masks
   
def train_model(): 
    """ 
    
    """
    model = unet()
    #Fitting and saving model
    model.fit(train_nparray, masks, batch_size=1, epochs=30, verbose=1, shuffle=True)
    model.save("model.h5")
    return None
   
def predict(test_nparray,model_path,save_result_path):
    """predict values and save then in a single numpy array."""
    #loading model and predicting mask
    model=unet()
    model.load_weights(model_path)

    prediction = model.predict(test_nparray, batch_size=4,verbose=1)
    np.save(save_result_path+'/prediction.npy', prediction)
    return prediction



def round_float_image(img): 
    
    img[img > 0.5 ] = 2
    img[img <= 0.5 ] = 0
  
    #changing to below one for multiclass

#     img[img > 1.0 ] = 2
#     img[img <= 1.0 ] = 0
    
    new_img=np.array(img,dtype=np.uint8)
    #change this carefully for multiclass unet.
    return new_img



def prepare_masks(result): 
    """converts an np array of (114,256,256,1) into proper masks...
    steps needed =
    1. convert np array into list of 114 np arrays
    2. resize to original size (maybe a different into the sequence.)
    2. Drop the 1 in the end
    3. Convert float to int type and convert values into 0,1 and 2s
    4. convert whatever is left to greyscale (it might be something different liek black and white??)
    
    """
    length,x,y,z=result.shape
    masks={}
    if(len(test_names)!=length):
        print("Length of result is not equal to test_names length")
    for i in range(len(test_names)): 
        img =result[i][:,:,0]
        #calling rounding function..and convert to uint8 there
        int_img=round_float_image(img)
        x,y=dict_videos_test[test_names[i]][str(0)].shape
        resized_img= reshape_image(int_img,y,x) # dont understand why need to put inversed x,y , but thats how it comes right.
        masks[test_names[i]]=resized_img
        #grescale part is left, check if it is really so!! by generating masks
    
    return masks


def write_masks(masks): 
    for k,v in masks.items():
        cv.imwrite('masks/' + k + '.png', v)



train_names=get_train_names()
test_names=get_test_names()




masks=preprocess_masks()



dict_videos_train=data_video_dictionary(train_names)
dict_videos_test=data_video_dictionary(test_names)



train_nparray=prepare_variance_data(dict_videos_train,train_names)
train_nparray.shape
test_nparray=prepare_variance_data(dict_videos_test,test_names) #same way of preparing both though..

test_nparray.shape


train_model()


prediction=predict(test_nparray,"model.h5",'result')


result=np.load('result/prediction.npy')
result.shape


np.unique(result[0])

masks=prepare_masks(result)
write_masks(masks)
