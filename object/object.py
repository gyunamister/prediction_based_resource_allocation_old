import sys
import os
from pathlib import Path

import numpy as np
import math

p = Path(__file__).resolve().parents[2]
sys.path.append(os.path.abspath(str(p)))


class Instance(object):
	#prediction 수행 -> predicted_next_act, time
	#prediction을 위한 input vector
	def __init__(self, name, weight, *args, **kwargs):
		super(Instance, self).__init__()
		if 'act_sequence' in kwargs:
			act_sequence = kwargs['act_sequence']
			self.set_act_sequence(act_sequence)
			#self.next_act = self.seq[0]
		elif 'initial_activity' in kwargs:
			self.next_act = kwargs['initial_activity']
		else:
			raise AttributeError('Either sequence or initial_activity should be given.')

		if 'res_sequence' in kwargs:
			res_sequence = kwargs['res_sequence']
			self.set_res_sequence(res_sequence)

		if 'dur_sequence' in kwargs:
			dur_sequence = kwargs['dur_sequence']
			self.set_dur_sequence(dur_sequence)

		if 'release_time' in kwargs:
			release_time = kwargs['release_time']
			self.set_release_time(release_time)
		else:
			release_time = False

		if 'initial_index' in kwargs:
			self.cur_index = kwargs['initial_index']
		else:
			self.cur_index = 0

		self.name = name
		self.next_actual_act = self.act_sequence[0]
		self.next_pred_act = self.act_sequence[0]
		self.next_actual_ts = release_time
		self.next_pred_ts = release_time
		self.weight=weight
		self.updated_weight=self.weight
		self.status = True
		self.pred_act_dur_dict = dict()
		self.res_sequence = list()

	def __str__(self):
		return self.name

	def __repr__(self):
		return self.name

	@classmethod
	def set_model(cls, pred_model):
		cls.pred_model = pred_model

	@classmethod
	def set_activity_list(cls, activity_list):
		cls.activity_list = activity_list

	@classmethod
	def set_resource_list(cls, resource_list):
		cls.resource_list = resource_list

	@classmethod
	def set_act_char_to_int(cls, act_char_to_int):
		cls.act_char_to_int = act_char_to_int

	@classmethod
	def set_act_int_to_char(cls, act_int_to_char):
		cls.act_int_to_char = act_int_to_char

	@classmethod
	def set_res_char_to_int(cls, res_char_to_int):
		cls.res_char_to_int = res_char_to_int

	@classmethod
	def set_res_int_to_char(cls, res_int_to_char):
		cls.res_int_to_char = res_int_to_char

	@classmethod
	def set_maxlen(cls, maxlen):
		cls.maxlen = maxlen

	def update_x(self, act_trace, res_trace):
		for i, act in enumerate(act_trace):
			activity_trace_i = act_trace[:i+1]
			resource_trace_i = res_trace[:i+1]
			act_int_encoded_X = [self.act_char_to_int[activity] for activity in activity_trace_i]
			res_int_encoded_X = [self.res_char_to_int[resource] for resource in resource_trace_i]

			# one hot encode X
			onehot_encoded_X = list()
			for act_value, res_value in zip(act_int_encoded_X, res_int_encoded_X):
				act_letter = [0 for _ in range(len(self.activity_list))]
				res_letter = [0 for _ in range(len(self.resource_list))]
				act_letter[act_value] = 1
				res_letter[res_value] = 1
				letter = act_letter + res_letter
				onehot_encoded_X.append(letter)
			num_act_res = len(self.activity_list)+len(self.resource_list)
			#zero-pad
			while len(onehot_encoded_X) != self.maxlen:
				onehot_encoded_X.insert(0, [0]*num_act_res)
		onehot_encoded_X = [onehot_encoded_X]
		onehot_encoded_X = np.asarray(onehot_encoded_X)
		return onehot_encoded_X

	def predict(self, pred_act, resource):
		act_trace = self.act_sequence[:self.cur_index] + [pred_act]
		res_trace = self.res_sequence[:self.cur_index] + [resource]
		X = self.update_x(act_trace, res_trace)
		act_pred, dur_pred = self.pred_model.predict(X)
		next_pred_act_index = np.argmax(act_pred,axis=1)[0]
		next_pred_act = self.act_int_to_char[next_pred_act_index]
		next_act_conf = np.max(act_pred)
		#pred_dur = int(dur_pred[0][0])
		pred_dur = math.ceil(dur_pred[0][0])
		if pred_dur == 0:
			pred_dur = 1
		return next_pred_act, next_act_conf, pred_dur

	def update_res_history(self, resource):
		self.res_sequence.append(resource)

	def get_name(self):
		return self.name

	def get_next_pred_act(self):
		return self.next_pred_act

	def get_next_act_conf(self):
		return self.next_act_conf

	def get_next_pred_ts(self):
		return self.next_pred_ts

	def get_next_ts_conf(self):
		return self.next_ts_conf

	def get_next_actual_ts(self):
		return self.next_actual_ts

	def get_next_actual_act(self):
		return self.next_actual_act

	def get_next_actual_act_dur(self):
		return self.next_actual_act_dur

	def get_pred_act_dur_dict(self):
		return self.pred_act_dur_dict

	def get_pred_act_dur(self, res):
		return self.pred_act_dur_dict[res]

	def get_release_time(self):
		return self.release_time

	def set_act_sequence(self, act_sequence):
		self.act_sequence = act_sequence

	def set_res_sequence(self, res_sequence):
		self.res_sequence = res_sequence

	def set_dur_sequence(self, dur_sequence):
		self.dur_sequence = dur_sequence

	def set_release_time(self, release_time):
		self.release_time = release_time

	def set_next_actual_act(self):
		if self.cur_index < len(self.act_sequence)-1:
			self.next_actual_act = self.act_sequence[self.cur_index]

	def set_next_actual_ts(self, next_actual_ts):
		self.next_actual_ts = next_actual_ts

	def set_actual_act_dur(self):
		if self.cur_index < len(self.act_sequence)-1:
			self.next_actual_act_dur = self.dur_sequence[self.cur_index]

	def set_pred_act_dur(self, res, pred_act_dur):
		self.pred_act_dur_dict[res] = pred_act_dur

	def clear_pred_act_dur(self):
		self.pred_act_dur_dict = dict()

	def set_next_pred_act(self, next_pred_act):
		self.next_pred_act = next_pred_act

	def set_next_act_conf(self, next_act_conf):
		self.next_act_conf = next_act_conf

	def set_next_ts_conf(self, next_ts_conf):
		self.next_ts_conf = next_ts_conf

	def set_next_pred_ts(self, next_pred_ts):
		self.next_pred_ts = next_pred_ts

	def update_index(self):
		#execute할 때 update
		self.cur_index += 1

	def update_weight(self,t):
		#기존에 ready instance에 있는 모든 instance에 대해
		self.updated_weight = self.weight + max(0,t-self.get_next_actual_ts())

	def get_weight(self):
		return self.updated_weight

	def reset_weight(self):
		#execute plan 할 때 초기화
		self.updated_weight = self.weight

	def set_status(self, status):
		self.status = status

	def get_status(self):
		return self.status

	def check_finished(self, t):
		if self.cur_index >= len(self.act_sequence):
			self.set_weighted_comp()
			return True
		else:
			#print("{}: current {}, goal {}".format(self.name, self.cur_index, len(self.act_sequence)))
			return False

	def set_weighted_comp(self):
		self.weighted_comp = (self.get_next_actual_ts()-self.release_time)*self.updated_weight

	def get_weighted_comp(self):
		return self.weighted_comp



class Resource(object):
	def __init__(self, name, skills, *args, **kwargs):
		super(Resource, self).__init__()
		self.name = name
		self.skills = skills
		self.next_pred_ts=0
		self.next_actual_ts = 0
		self.next_ts_conf = 1
		self.status=True
		self.dur_dict = dict()

	def __str__(self):
		return self.name

	def __repr__(self):
		return self.name

	def get_name(self):
		return self.name

	def get_skills(self):
		return self.skills

	def get_next_pred_ts(self):
		return self.next_pred_ts

	def get_next_ts_conf(self):
		return self.next_ts_conf

	def get_next_actual_ts(self):
		return self.next_actual_ts

	def set_next_actual_ts(self, next_actual_ts):
		self.next_actual_ts = next_actual_ts

	def set_next_ts_conf(self, next_ts_conf):
		self.next_ts_conf = next_ts_conf

	def set_next_pred_ts(self, next_pred_ts):
		self.next_pred_ts = next_pred_ts

	def set_status(self, status):
		self.status = status

	def get_status(self):
		return self.status



