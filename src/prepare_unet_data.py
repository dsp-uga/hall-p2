
import preprocess_unet as prunet
import unet_model as cnn


def train_unet_model(data,mask_list):
    train_data = prunet.shape_img_data(data)
    masks = prunet.shape_mask_data(mask_list)
    model = cnn.unet()
    #Fitting and saving model
    model.fit(train_data, masks, batch_size=2, epochs=15, verbose=1, shuffle=True)
    model.save("../dataset/model.h5")

def test_unet_model(data):
    test_data = shape_img_data(data)
    model = cnn.unet()
    model.load_weights('../dataset/model.h5')

    prediction = model.predict(test_data, batch_size =4, verbose = 1)
    np.save('../dataset/prediction.npy', prediction)
