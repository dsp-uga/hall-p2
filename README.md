# hall-p2

## Ciliary Motion Extraction : Cilia Segmentation
Cilia is a hair-like object protruding out of cell-bodies. Our task is to segment them and identify regions with Celia. This problem is particularly hard and lack of data makes it even harder. We only had 214 video instance to train.  
The core aim of a learning algorithim in Cilia segmentation should be to learn texture of Celia and how it moves in a video. 

This repository takes small video clips of cilla and returns segmetations of the frames to identify the location of the cillia. This project was created for CSCI 8360 Data Science Practicum at the University of Georgia. 

This project took different approaches to identify the cillia with fluctuation variance, optical flow, and unets. 
  * Fluctuation Variance
    * We compute the variance across frames. This will tell us how much each pixel has changed compared to the mean, but also will help us identify regions that could be Cillia. 
  * Optical Flow
    * This approach looks at two frames at a time to find movement of objects between them. It creates a 2D vector that shows the displacement of each pixel from frame to frame. 
    * Opitcal Flow has three assumptions: 
      1. The pixel intensities of objects does not change between consecutive frames
      2. Surrounding pixels have similar motion
      3. Due to these assumptions, we are assuming for this project that any detections of movement in the frames will be Cilla
    * Due to these assumptions, we can solve for the change in postion given some change in time, which is what optical flow is. Looking at how much the postion changed in respect to left/right and also up/down. There are different methods of finding the remaining variables of the equation, which we are using the Dense Optical Flow method, using the Gunner Farneback's algorithm. These different methods are implemented in the package [OpenCV](https://opencv.org/). This method takes the optical flow for all the points in the frame, where other methods look at specific regions of the frames. 
    * This method used different preprocessing methods before the Optical Flow was calculated. We mean centered, finiding the mean of an image and subtracting it from each pixel, and or normalizing the images. 
  * UNets
  UNets have proved to be highly efficient and effective in biomedical imaging domain. They don't require as much data as other CNN architectures such as FCNs do. 
We implement Unets from [this github repository](https://github.com/zhixuhao/unet). It is based on the model built by researchers who invented Unets. 

### Getting Started
These next two sections will help you run this project on your local machine to attempt at replicating our results. 

#### Prerequisites
This project uses different Python packages listed below:
  * [Keras](https://keras.io/): A deep learning package that is used to build the UNets that really makes the code easy to follow.
  * [OpenCv](https://opencv.org/): The library that contains different methods to processes images and implements the Optical Flow functions
  * [Skimage](https://scikit-image.org/): Another image processing library that we used for different prepressing [filters](http://scikit-image.org/docs/dev/api/skimage.filters.html). We used these 4 filters: roberts, sobel, scharr, and prewitt. 
 * [Tensorflow](https://www.tensorflow.org/)  
 * [GPU setup](https://medium.com/@raza.shahzad/setting-up-tensorflow-gpu-keras-in-conda-on-windows-10-75d4fd498198). We used this link to setup GPU to train unets using Keras and tensorflow backend. 
#### Installing Dependencies
[Conda](https://conda.io/en/latest/) will easily mange the enviornment and install all dependencies for these libraries that we used. 

### Built With
  * [Python 3.7](https://www.python.org/)

### Authors
The list of authors and their contributions are listed here -- [Contributors](CONTRIBUTORS.md)
### License
This project is licensed under the MIT Liscense -- [License](LICENSE)
