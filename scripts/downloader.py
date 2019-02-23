import os

 
if not os.path.exists('../dataset'):
    os.makedirs('../dataset')
	
def text_download():

	command = 'gsutil cp gs://uga-dsp/project2/test.txt'+' ../dataset/'
	os.system(command)
	command2 = 'gsutil cp gs://uga-dsp/project2/train.txt'+' ../dataset/'
	os.system(command2)

 

def data_download():

	if not os.path.exists('../dataset/data'):
		os.makedirs('../dataset/data')

	command = 'gsutil cp gs://uga-dsp/project2/data/*'+' ../dataset/data/'
	os.system(command)

def masks_download():

	if not os.path.exists('../dataset/masks'):
		os.makedirs('../dataset/masks')

	command = 'gsutil cp gs://uga-dsp/project2/masks/*'+' ../dataset/masks/'
	os.system(command)


text_download()
data_download()
masks_download()


