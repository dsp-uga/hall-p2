import os


#if there is no dataset folder, creating one.  
if not os.path.exists('../dataset'):
    os.makedirs('../dataset')
	
def text_download():
	"""
	downloads train.txt and test.txt files from google cloud links : gs://uga-dsp/project2/test.txt
	"""
	
	command = 'gsutil cp gs://uga-dsp/project2/test.txt'+' ../dataset/'
	#commands are executed
	os.system(command)
	
	command2 = 'gsutil cp gs://uga-dsp/project2/train.txt'+' ../dataset/'
	os.system(command2)

 

def data_download():
	"""
	Downloads data folder from google cloud , which contains folders for 300+ video frames of Celia
	"""
	#if data folder does not exist, it is created
	if not os.path.exists('../dataset/data'):
		os.makedirs('../dataset/data')

	command = 'gsutil cp gs://uga-dsp/project2/data/*'+' ../dataset/data/'
	os.system(command)



def masks_download():
	"""
	Downloads masks for 210+ train images , which are used as labels for training 
	"""
	#if masks folder does not exist, it is created. 
	if not os.path.exists('../dataset/masks'):
		os.makedirs('../dataset/masks')

	command = 'gsutil cp gs://uga-dsp/project2/masks/*'+' ../dataset/masks/'
	os.system(command)

#executing methods 
text_download()
data_download()
masks_download()


