import sys
import os
import signal
import pandas as pd
import math
import datetime
import numpy as np
from datetime import datetime
from pathlib import Path
import copy
import random

from keras.layers import LSTM
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Nadam

p = Path(__file__).resolve().parents[2]
sys.path.append(os.path.abspath(str(p)))
from PyProM.src.data.Eventlog import Eventlog


def extract_checkpoints(patient_ready_time):
	return min(patient_ready_time.values())

def preparation(path):
	eventlog = Eventlog.from_txt(path, sep=',')
	eventlog = eventlog.assign_caseid('CASE_ID')
	eventlog = eventlog.assign_activity('Activity')
	eventlog = eventlog.assign_resource('Resource')
	#eventlog = eventlog.assign_timestamp('Start Timestamp', name='Timestamp', format = '%Y/%m/%d %H:%M:%S')
	#eventlog = eventlog.clear_columns()

	#eventlog['Duration'] = eventlog['Complete'] - eventlog['Start']
	return eventlog

def train_test_split(eventlog, train_ratio, test_ratio):
	caseid = list(eventlog.get_caseids())
	num_train = int(len(caseid)*train_ratio)
	num_test = len(caseid)*test_ratio
	train_caseid = list(random.sample(caseid, num_train))
	test_caseid = [x for x in caseid if x not in train_caseid]
	train = eventlog.loc[eventlog['CASE_ID'].isin(train_caseid)]
	test = eventlog.loc[eventlog['CASE_ID'].isin(test_caseid)]
	return train, test

def preprocess(eventlog):
	print(eventlog)
	activity_trace = eventlog.get_event_trace(4,'Activity')
	duration_trace = eventlog.get_event_trace(4,'Duration')
	resource_trace = eventlog.get_event_trace(4,'Resource')
	activity_list = sorted(list(eventlog.get_activities()))
	activity_list.append('!')

	resource_list = sorted(list(eventlog.get_resources()))


	# define a mapping of chars to integers
	act_char_to_int = dict((c, i) for i, c in enumerate(activity_list))
	act_int_to_char = dict((i, c) for i, c in enumerate(activity_list))

	res_char_to_int = dict((c, i) for i, c in enumerate(resource_list))
	res_int_to_char = dict((i, c) for i, c in enumerate(resource_list))
	# integer encode input data
	X_train = list()
	y_a = list()
	y_t = list()
	trace_len = [len(x) for x in activity_trace.values()]

	res_trace_len = [len(x) for x in resource_trace.values()]

	maxlen = max(trace_len) + max(res_trace_len)
	num_act_res = len(activity_list) + len(resource_list)
	num_act = len(activity_list)

	for case in activity_trace:
		for i, trace in enumerate(activity_trace[case]):
			activity_trace_i = activity_trace[case][:i+1]
			resource_trace_i = resource_trace[case][:i+1]
			act_int_encoded_X = [act_char_to_int[activity] for activity in activity_trace_i]
			res_int_encoded_X = [res_char_to_int[resource] for resource in resource_trace_i]

			#해당하는 next_resource의 next_duration을 예측하는 모델을 만드는 것임
			"""
			if i==len(activity_trace[case])-1:
				res_letter = [0 for _ in range(len(resource_list))]
			else:
				next_resource = resource_trace[case][i+1]
				int_encoded_next_resource = res_char_to_int[next_resource]
				res_letter = [0 for _ in range(len(resource_list))]
				res_letter[int_encoded_next_resource] = 1
			"""

			# one hot encode X
			onehot_encoded_X = list()
			for act_value, res_value in zip(act_int_encoded_X, res_int_encoded_X):
				act_letter = [0 for _ in range(len(activity_list))]
				res_letter = [0 for _ in range(len(resource_list))]
				act_letter[act_value] = 1
				res_letter[res_value] = 1
				letter = act_letter + res_letter
				onehot_encoded_X.append(letter)
			#zero-pad
			while len(onehot_encoded_X) != maxlen:
				onehot_encoded_X.insert(0, [0]*num_act_res)
			if len(onehot_encoded_X) > maxlen:
				print(onehot_encoded_X)
			X_train.append(onehot_encoded_X)

			# one hot encode y
			if i==len(activity_trace[case])-1:
				next_act = '!'
			else:
				next_act = activity_trace[case][i+1]
			current_duration = duration_trace[case][i]
			int_encoded_next_act = act_char_to_int[next_act]
			letter = [0 for _ in range(len(activity_list))]
			letter[int_encoded_next_act] = 1
			y_a.append(letter)
			y_t.append(current_duration)

	X_train = np.asarray(X_train)
	y_a = np.asarray(y_a)
	y_t = np.asarray(y_t)
	#X_train = X_train.reshape(X_train.shape[0], maxlen, num_act)
	#y_a = y_a.reshape(y_a.shape[0], num_act)
	return X_train, y_a, y_t, maxlen, num_act, num_act_res

if __name__ == '__main__':
	eventlog = preparation(path="../result/traininglog_0121_2.csv")
	#eventlog = preparation(path='../../result/modi_BPI_2012_dropna_filter_act.csv')
	train_ratio=0.7
	test_ratio=0.3
	train, test = train_test_split(eventlog, train_ratio, test_ratio)

	X_train, y_a, y_t, maxlen, num_act, num_act_res = preprocess(train)
	X_test, y_a_test, y_t_test, maxlen, num_act, num_act_res = preprocess(test)


	#model = Sequential()
	main_input = Input(shape=(maxlen, num_act_res), name='main_input')
	l1 = LSTM(100, kernel_initializer='glorot_uniform', return_sequences=True, dropout_W=0.2)(main_input)
	b1 = BatchNormalization()(l1)

	l2_1 = LSTM(100, kernel_initializer='glorot_uniform', return_sequences=False, dropout_W=0.2)(b1) # the layer specialized in activity prediction
	b2_1 = BatchNormalization()(l2_1)
	l2_2 = LSTM(100, kernel_initializer='glorot_uniform', return_sequences=False, dropout_W=0.2)(b1) # the layer specialized in time prediction
	b2_2 = BatchNormalization()(l2_2)
	act_output = Dense(num_act, activation='softmax',kernel_initializer='glorot_uniform', name='act_output')(b2_1)
	time_output = Dense(1, kernel_initializer='glorot_uniform', name='time_output')(b2_2)

	model = Model(input=[main_input], output=[act_output, time_output])
	opt = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004, clipvalue=3)
	model.compile(loss={'act_output':'categorical_crossentropy', 'time_output':'mae'}, optimizer=opt)
	early_stopping = EarlyStopping(monitor='val_loss', patience=42)
	#model.fit(X_train, y_a, validation_split=0.2, epochs=100, batch_size=maxlen, verbose=2, callbacks=[early_stopping])
	model.fit(X_train, {'act_output':y_a, 'time_output':y_t}, validation_split=0.2, verbose=2, callbacks=[early_stopping], batch_size=maxlen,epochs=10)
	# serialize model to JSON
	model_json = model.to_json()
	#with open("BPI_2012_model.json", "w") as json_file:
	with open("test_model_0121_1.json", "w") as json_file:
	    json_file.write(model_json)
    # serialize weights to HDF5
	model.save_weights("test_model_0121_1.h5")
	print("Saved model to disk")
	# Final evaluation of the model
	scores = model.evaluate(X_test, {'act_output':y_a_test, 'time_output':y_t_test} , verbose=0)
	print(scores)
	print(model.metrics_names)
	#print("Accuracy: %.2f%%" % (scores[1]*100))