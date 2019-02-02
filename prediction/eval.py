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
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Nadam

p = Path(__file__).resolve().parents[3]
sys.path.append(os.path.abspath(str(p)))
from PyProM.src.data.Eventlog import Eventlog

def extract_checkpoints(patient_ready_time):
	return min(patient_ready_time.values())

def preparation(path="../result/generatedlog1.csv"):
	eventlog = Eventlog.from_txt(path, sep=',')
	eventlog = eventlog.assign_caseid('CaseID')
	eventlog = eventlog.assign_activity('Activity')
	eventlog = eventlog.assign_resource('Resource')
	#eventlog = eventlog.assign_timestamp('Start Timestamp', name='Timestamp', format = '%Y/%m/%d %H:%M:%S')
	#eventlog = eventlog.clear_columns()

	eventlog['Duration'] = eventlog['Complete'] - eventlog['Start']
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
	activity_trace = eventlog.get_event_trace(4,'ACTIVITY')
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
	eventlog = preparation(path="../../result/generatedlog1.csv")
	train_ratio=0.7
	test_ratio=0.3
	train, test = train_test_split(eventlog, train_ratio, test_ratio)

	#X_train, y_train, maxlen, num_act = preprocess(train)
	X_test, y_a, y_t, maxlen, num_act, num_act_res = preprocess(test)
	print(X_test.shape)
	print(X_test[0])
	print(X_test[1])

	from keras.models import model_from_json
	json_file = open("./second_model.json", "r")
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)

	loaded_model.load_weights("./second_model.h5")
	print("Loaded model from disk")

	opt = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004, clipvalue=3)
	loaded_model.compile(loss={'act_output':'categorical_crossentropy', 'time_output':'mae'}, optimizer=opt)

	sample = X_test[:5][:][:]
	print(sample.shape)
	#sample = [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]]]

	sample = np.asarray(sample)
	sample_ans_a = y_a[:5][:][:]
	sample_ans_t = y_t[:5][:][:]
	print(sample,sample_ans_a, sample_ans_t)
	print(loaded_model.predict(sample))
	"""
	#model = Sequential()
	main_input = Input(shape=(maxlen, num_act), name='main_input')
	l1 = LSTM(100, init='glorot_uniform', dropout_W=0.2)(main_input)
	b1 = BatchNormalization()(l1)
	act_output = Dense(num_act, activation='softmax',init='glorot_uniform', name='act_output')(b1)

	model = Model(input=[main_input], output=[act_output])
	opt = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004, clipvalue=3)
	model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
	early_stopping = EarlyStopping(monitor='val_loss', patience=42)
	model.fit(X_train, y_train, validation_split=0.2, epochs=10, batch_size=maxlen, verbose=2, callbacks=[early_stopping])
	# serialize model to JSON
	model_json = model.to_json()
	with open("new_model.json", "w") as json_file:
	    json_file.write(model_json)
    # serialize weights to HDF5
	model.save_weights("new_model.h5")
	print("Saved model to disk")
	# Final evaluation of the model
	scores = model.evaluate(X_test, y_test, verbose=0)
	print("Accuracy: %.2f%%" % (scores[1]*100))
	"""

