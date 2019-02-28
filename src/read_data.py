def read_filename(filepath):
    '''
    This function takes in as input full path to the directory having
    the names of the files to be read and returns a list of all the filenames.
    '''
    #initializing a list
    files = []
    #opening the file in read mode
    f = open(filepath, 'r')
    for file in f:
        # appending each filename in the list
        files.append(file.strip())
    return files

def load_video(videopath):
    '''
    This function is called by load_whole_data and takes in as input path to the videos.
    Appends each of the frames to a list and returns the final list.
    '''
    #initializing the list to store videos
    video = []

    for filename in os.listdir(videopath):
        #reading the image at the location videopath
        img = cv.imread(os.path.join(videopath,filename),0)
        if img is not None:
            #appending each image to the list.
            video.append(img)
    return video

def load_whole_data(directory, file_list):
    '''
    This function takes in as input path to the directory containing videos
    and a list of name of the files to be read. It creates a list of list of each of the data
    folder and the frames in it. Returns this list of list.
    '''
    #initializing the list to store the data
    data = []
    #looping through the filenames
    for i in file_list:
        #adding the filename to the directory path
        datapath = os.path.join(directory, i)
        #calling the load video to get all the frames of the file and appending it to the list
        data.append(load_video(datapath))
    return data

def load_mask(mask_path,file_list):
    '''
    This function takes as input the path to the directory where masks are stored and
    name of the files whose mask is to be read. Returns a list containing masks for each of the files.
    '''
    #initializing list to store the masks
    mask = []
    #looping through the files
    for i in file_list:
        #reading the mask and appending it to the list
        mask.append(cv.imread(os.path.join(mask_path,i)+ '.png' ,0))
    return mask
