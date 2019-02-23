import os
import tarfile

def extract_tars():
#creates new folder data within data and extracts all tar files into it
	for filename in os.listdir('../dataset/data'):
		print(filename)
		tar=tarfile.open('../dataset/data/'+filename)
		tar.extractall('../dataset')
		tar.close()
		#deletes tar file as we have extracted images from it
		os.remove('../dataset/data/'+filename)

	
extract_tars()
