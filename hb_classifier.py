from numpy import genfromtxt
import numpy as np 
from time import time
import matplotlib.pyplot as plt 
import random
import os
import sys
from util import*

'''
heartbeat classification using convolution neural network
@author: Kemeng Chen: kemengchen@email.arizona.edu
'''

def beat_classify(file_name):
	'''
	Heartbeat classification using convolution network
	beat length (sampling rate) needs to be scaled since model was 
	trained using ECG of 360Hz
	'''
	fs=360
	beat_lgth=201
	prior=80
	post=120
	model_name='cnn_hb_model.meta'
	model_path=os.path.join(os.getcwd(), 'models')
	file_path=os.path.join(os.getcwd(), 'data', file_name)
	if not os.path.isfile(file_path):
		raise ValueError(file_path, 'not found')
	ecg=genfromtxt(file_path)
	hb_model=restore_model(model_path, model_name)
	ts=time()

	R_peaks, _, _=EKG_QRS_detect(ecg, fs, False, False)
	beat_matrix=get_beat(ecg, R_peaks, prior, post)
	predication=hb_model.run_model(beat_matrix)

	te=time()
	print('Time cost: ',str(te-ts))

	# feature_map_table={1:'Normal',\
	#          	       2:'Aberrated atrial premature',\
	#                    3:'Atrial premature',\
	#                    4:'Ventricular escape',\
	#                    5:'Nodal escape',\
	#                    6:'Nodal premature',\
	#                    7:'Left bundle branch block',\
	#                    8:'Right bundle branch block'}
	feature_map_table={1:'Normal',\
	         	       2:'AAP',\
	                   3:'AP',\
	                   4:'VE',\
	                   5:'NE',\
	                   6:'NP',\
	                   7:'LB',\
	                   8:'RB'}

	index=np.arange(ecg.shape[0])/fs
	plt.style.use('seaborn-bright')
	fig, axes=plt.subplots()
	axes.plot(index, ecg)
	axes.plot(R_peaks/fs, ecg[R_peaks], 'ro')
	axes.set_title('ECG heartbeat label')
	axes.set_xlabel('Time [sec]')
	axes.tick_params(direction='in', length=1)
	axes.set_xlim([index[0], index[index.shape[0]-1]])
	axes.set_ylim([np.amin(ecg)-20, np.amax(ecg)+50])
	axes.grid()

	predict=predication[0]
	for i in range(R_peaks.shape[0]):
		temp_index=(i*beat_lgth+prior)/fs
		peak=ecg[i*beat_lgth+prior]+30
		axes.text(temp_index, peak, feature_map_table[predict[i]+1], color='g')
	plt.show()

def get_beat(ecg, R_peaks, prior, post):
	num_peak=R_peaks.shape[0]
	beat_matrix=np.zeros([num_peak, prior+post+1])
	for index in range(num_peak):
		peak=R_peaks[index]
		temp=ecg[peak-prior:peak+post+1] 
		beat_matrix[index,:]=temp
	return beat_matrix

if __name__ == '__main__':
	if len(sys.argv)<2:
		raise ValueError('No file name specified')
	beat_classify(sys.argv[1])