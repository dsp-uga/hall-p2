
import preprocess_unet as prunet
import unet_model as cnn


def train_unet_model(data,mask_list):
    '''
    This function takes as input the data as list of list 
    and mask as the list and train the unet model after some preprocessing on the data
    '''
    train_data = prunet.shape_img_data(data) # calling the function to bring the data to a shape to be passsed for training
    masks = prunet.shape_mask_data(mask_list) # calling the function bring the mask to a shape to be passed for training.
    model = cnn.unet() 
    #Fitting and saving model
    model.fit(train_data, masks, batch_size=2, epochs=15, verbose=1, shuffle=True) # fitting the model on the training data
    model.save("../dataset/model.h5") #saving the model created.

def test_unet_model(data):
    '''
    This function takes as input the test data as list of list. Preprocess this data to bring it to
    the desired shape for prediction and performs predictions
    '''
    test_data = shape_img_data(data) # calling the function to bring the data to the desired shape for prediction.
    model = cnn.unet()
    model.load_weights('../dataset/model.h5') #loading the model created after training

    prediction = model.predict(test_data, batch_size =4, verbose = 1) #performing predictions
    np.save('../dataset/prediction.npy', prediction) # saving the predictions
 
