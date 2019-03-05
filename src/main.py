import os
import tarfile
import numpy as np
import argparse 
import downloader as dld
import tar_extractor as ext
import preproc as pre
import read_data as pre
import read_data as read
import optical_flow as opt

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',type=str,help='train or test')
    parser.add_argument('--preproc_type',type=str,help='full or mean  or normalize')
    parser.add_argument('--optical_flow',type=str,help='full or step_wise  or first_two')
    parser.add_argument('--image_processing',type=str,help='none or sobel or roberts or prewitt or kirsch')
    parser.add_argument('--unet',type=str,help='T or F')
    args = parser.parse_args()


    #downloading the data as tar files
    dld.download_data()

    #calling extractor to extract the downloaded files
    ext.extract_tars()

    if args.mode =='train':
        #reading the names of the files in a list
        file_list = read.read_filename('../dataset/train.txt')

        #reading the whole data as list of list containing all frames for each file
        whole_data = read.load_whole_data('../dataset/data', file_list)

        #reading the masks in the list
        mask_list = read.load_mask('../dataset/masks', file_list)

    if args.mode =='test':
        #reading the names of the files in a list
        file_list = read.read_filename('../dataset/test.txt')

        #reading the whole data as list of list containing all frames for each file
        whole_data = read.load_whole_data('../dataset/data', file_list)

    if args.preproc_type == 'all':

        mean_center_data = pre.mean_center(whole_data) #calling the function to mean center all the images in the data.
        np.save('../dataset/mean_center_data.npy', mean_data) # writing the data to disk
        del mean_center_data #deleting the  data from memory after writing




        normal_data = pre.noramlize_img(whole_data) #calling the function to normalize each of the frames
        np.save('../dataset/normal_data.npy') #writing the data to disk
        del normal_data #deleting the  data from memory after writing

    elif args.preproc_type == 'mean':
        mean_center_data = pre.mean_center(whole_data) #calling the function to mean center all the images in the data.
        np.save('../dataset/mean_center_data.npy', mean_data) #writing the mean data to the disk
        del mean_center_data #deleting the  data from memory after writing

    elif args.preproc_type == 'normalize':
        normal_data = pre.noramlize_img(whole_data) #calling the function to normalize each of the frames
        np.save('../dataset/normal_data.npy') #writing the data to disk
        del normal_data #deleting the  data from memory after writing

    print('\n before running optical flow')

    opt_flow_data = []
    for i in range(0, len(whole_data)):
        opt_flow_data.append(opt.optical_flow(whole_data[i],args.optical_flow, args.image_processing))


    print('\n before saving optical flow to disc')
    np.save('../dataset/opflow.npy',np.array(opt_flow_data)) # saving the data to the disk
    del opt_flow_data # deleting the data from the memory

