# hall-p2
Cilia Segmentation

This repository takes small video clips of cilla and returns segmetations of the frames to basically identify the location of the cillia. This project was createdfor CSCI 8360 Data Science Practicum at the University of Georgia. 

This project took different approaches to identify the cillia with fluctuation variance, optical flow, and unets. 
  * Fluctuation Variance
    * We take the frames of each video and compute the variance. This will tell us how much each pixel has changed compared to the mean, but also will help us identify regions that could be Cillia. 
  * Optical Flow
    * This approach looks at two frames at a time to find movement of objects between them. It creates a 2D vector that shows the displacement of each pixel from frame to frame.
    * Opitcal Flow has three assumptions: 
      1. The pixel intensities of objects does not change between consecutive frames
      2. Surrounding pixels have similar motion
      3. Due to these assumptions, we are assuming for this project that any detections of movement in the frames will be Cilla
    * Due to these assumptions, we can solve for the change in postion given some change in time, which is what optical flow is. Looking at how much the postion changed in respect to left/right and also up/down. There are different methods of finding the remaining variables of the equation, which we are using the Dense Optical Flow method, using the Gunner Farneback's algorithm. These different methods are implemented in the package [OpenCV](https://opencv.org/). This method takes the optical flow for all the points in the frame, where other methods look at specific regions of the frames. 
  * UNets
### Getting Started

#### Prerequisites

#### Installing Dependencies

### Built With


### Contributing

### Authors

### License
