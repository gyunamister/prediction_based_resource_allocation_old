import sys
import os
from pathlib import Path
import networkx as nx
import time
import numpy as np

p = Path(__file__).resolve().parents[1]
sys.path.append(os.path.abspath(str(p)))

from PyProM.src.data.Eventlog import Eventlog
from object.object import Instance, Resource


class BaseOptimizer(object):
	def __init__(self, *args, **kwargs):
		super(BaseOptimizer, self).__init__(*args, **kwargs)
		self.w_comp_time = list()

	def load_data(self,path):
		eventlog = Eventlog.from_txt(path, sep=',')
		eventlog = eventlog.assign_caseid('CASE_ID')
		eventlog = eventlog.assign_activity('Activity')
		eventlog = eventlog.assign_resource('Resource')
		return eventlog

	def load_real_data(self,path):
		eventlog = Eventlog.from_txt(path, sep=',')
		eventlog = eventlog.assign_caseid('CASE_ID')
		eventlog = eventlog.assign_activity('Activity')
		eventlog['Resource'] = eventlog['Resource'].astype(int)
		eventlog = eventlog.assign_resource('Resource')
		eventlog = eventlog.assign_timestamp(name='StartTimestamp', new_name='StartTimestamp', _format = '%Y.%m.%d %H:%M:%S', errors='raise')

		def to_minute(x):
			t = x.time()
			minutes = t.hour * 60 + t.minute
			return minutes

		eventlog['Start'] = eventlog['StartTimestamp'].apply(to_minute)
		return eventlog

	def initialize_real_instance(self, eventlog):
		instance_set = list()
		activity_trace = eventlog.get_event_trace(workers=4, value='Activity')
		resource_trace = eventlog.get_event_trace(4,'Resource')
		date_trace = eventlog.get_event_trace(workers=4, value='StartDate')
		time_trace = eventlog.get_event_trace(workers=4, value='Start')
		dur_trace = eventlog.get_event_trace(workers=4, value='Duration')
		weight_trace = eventlog.get_event_trace(workers=4, value='weight')
		for i, case in enumerate(activity_trace):
			#release_time = 0
			weight = min(weight_trace[case])
			initial_index=0
			for j, time in enumerate(date_trace[case]):
				if time == '2012-03-10':
					initial_index =j-1
					release_time = time_trace[case][j]
					break
			instance = Instance(name=case, weight=weight, release_time=release_time, act_sequence=activity_trace[case], res_sequence=resource_trace[case],dur_sequence=dur_trace[case], initial_index=initial_index)
			instance_set.append(instance)
		return instance_set

	def initialize_test_instance(self, eventlog):
		instance_set = list()
		activity_trace = eventlog.get_event_trace(workers=4, value='Activity')
		resource_trace = eventlog.get_event_trace(4,'Resource')
		time_trace = eventlog.get_event_trace(workers=4, value='Start')
		dur_trace = eventlog.get_event_trace(workers=4, value='Duration')
		weight_trace = eventlog.get_event_trace(workers=4, value='weight')
		for case in activity_trace:
			release_time = min(time_trace[case])
			release_time = 0
			weight = min(weight_trace[case])
			instance = Instance(name=case, weight=weight, release_time=release_time, act_sequence=activity_trace[case], res_sequence=resource_trace[case],dur_sequence=dur_trace[case])
			instance_set.append(instance)
		return instance_set

	def initialize_real_resource(self, eventlog, test_log):
		resource_set = list()
		resource_list = sorted(list(test_log.get_resources()))
		#resource_list = [str(x) for x in resource_list]

		for res in resource_list:
			act_list = list(test_log.loc[test_log['Resource']==res,'Activity'].unique())
			resource = Resource(res, act_list)
			resource_set.append(resource)
		"""
		for res in resource_set:
			print(res.get_name(), res.get_skills())
		"""
		#time.sleep(3)
		return resource_set

	def initialize_test_resource(self, eventlog):
		resource_set = list()
		resource_list = sorted(list(eventlog.get_resources()))
		for res in resource_list:
			act_list = list(eventlog.loc[eventlog['Resource']==res,'Activity'].unique())
			resource = Resource(res, act_list)
			resource_set.append(resource)
		return resource_set

	def set_basic_info(self, eventlog):
		activity_trace = eventlog.get_event_trace(4,'Activity')
		resource_trace = eventlog.get_event_trace(4,'Resource')
		activity_list = sorted(list(eventlog.get_activities()))
		activity_list.append('!')
		resource_list = sorted(list(eventlog.get_resources()))

		act_char_to_int = dict((c, i) for i, c in enumerate(activity_list))
		act_int_to_char = dict((i, c) for i, c in enumerate(activity_list))

		res_char_to_int = dict((c, i) for i, c in enumerate(resource_list))
		res_int_to_char = dict((i, c) for i, c in enumerate(resource_list))

		trace_len = [len(x) for x in activity_trace.values()]
		res_trace_len = [len(x) for x in resource_trace.values()]
		maxlen = max(trace_len) + max(res_trace_len)
		Instance.set_activity_list(activity_list)
		Instance.set_resource_list(resource_list)
		Instance.set_act_char_to_int(act_char_to_int)
		Instance.set_act_int_to_char(act_int_to_char)
		Instance.set_res_char_to_int(res_char_to_int)
		Instance.set_res_int_to_char(res_int_to_char)
		Instance.set_maxlen(maxlen)

	def set_model(self, json_file_path, model_file_path):
		#예측 모델 로드
		from keras.models import model_from_json
		json_file = open(json_file_path, "r")
		loaded_model_json = json_file.read()
		json_file.close()
		loaded_model = model_from_json(loaded_model_json)

		loaded_model.load_weights(model_file_path)
		print("Loaded model from disk")
		from keras.optimizers import Nadam
		opt = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004, clipvalue=3)
		loaded_model.compile(loss={'act_output':'categorical_crossentropy', 'time_output':'mae'}, optimizer=opt)
		return loaded_model



	def prepare_test(self, test_path, json_file_path, model_file_path):
		pred_model = self.set_model(json_file_path, model_file_path)
		Instance.set_model(pred_model)
		eventlog = self.load_data(path=test_path)
		#resource set 생성 - 어떤 resource가 있고 어떤 acitivity 수행 가능한지 (phase 1에서 dict 형태로 생성)
		resource_set = self.initialize_test_resource(eventlog)


		#inital instance 생성
		#set_release_schedule를 참고하여 생성하면 될 듯
		#즉, release_schedule return
		instance_set = self.initialize_test_instance(eventlog)
		#prediction을 위한 X 생성을 위해 Instance의 attribute를 생성
		self.set_basic_info(eventlog)
		return resource_set, instance_set

	def prepare_real(self, test_path, org_log_path, json_file_path, model_file_path):
		pred_model = self.set_model(json_file_path=json_file_path, model_file_path=model_file_path)
		Instance.set_model(pred_model)
		eventlog = self.load_real_data(path=org_log_path)

		test_log = self.load_real_data(path=test_path)
		test_log['weight'] = test_log['weight']

		instance_set = self.initialize_real_instance(test_log)
		resource_set = self.initialize_real_resource(eventlog, test_log)
		self.set_basic_info(eventlog)
		return resource_set, instance_set


	def update_ongoing_instances(self, instance_set, ongoing_instance, t):
		for i in instance_set:
			if i.get_release_time() == t:
				ongoing_instance.append(i)
		return ongoing_instance

	def update_object(self, ongoing_instance, resource_set, t):
		for j in resource_set:
			if j.get_next_actual_ts() <= t:
				j.set_status(True)

		for i in ongoing_instance:
			if i.get_next_actual_ts() <= t:
				i.set_status(True)
		ready_instance = [x for x in ongoing_instance if x.get_status()==True]
		ready_resource = [x for x in resource_set if x.get_status()==True]
		G = nx.DiGraph()
		for i in ready_instance:
			actual_act = i.get_next_actual_act()
			i.clear_pred_act_dur()
			for j in ready_resource:
				if actual_act in j.get_skills():
					G.add_edge('s',i, capacity=1)
					G.add_edge(j,'t',capacity=1)
					weight = i.get_weight()
					cost = weight * (-1)
					G.add_edge(i,j,weight=cost,capacity=1)

		return G


	def update_plan(self, G,t):
		#if 's' in nodes and 't' in nodes:
		nodes=G.nodes()
		if len(nodes)!=0:
			M = nx.max_flow_min_cost(G, 's', 't')
		else:
			M=False
		#print(M)
		return M


	def execute_plan(self, ongoing_instance, resource_set, M, completes, t):
		ready_instance = [x for x in ongoing_instance if x.get_status()==True]
		ready_resource = [x for x in resource_set if x.get_status()==True]
		#print("ongoing: {}".format(ongoing_instance))
		#print("ready: {}".format(ready_instance))
		if M!=False:
			for i in M:
				if i in ready_instance:
					for j, val in M[i].items():
						if val==1 and j in ready_resource:
							#act_dur 유지
							i.clear_pred_act_dur()
							next_actual_act = i.get_next_actual_act()
							next_pred_act, next_act_conf, pred_dur = i.predict(next_actual_act, j.get_name())
							#i.set_next_pred_act(next_pred_act)
							#i.set_next_act_conf(next_act_conf)
							i.set_next_pred_ts(t+pred_dur)

							i.update_index()
							#i.update_weight(t)
							i.set_next_actual_act()
							#i.set_actual_act_dur()
							i.set_next_actual_ts(i.get_next_pred_ts())

							j.set_next_pred_ts(t+pred_dur)
							j.set_next_actual_ts(j.get_next_pred_ts())

							i.set_status(False)
							j.set_status(False)

							i.update_res_history(j.get_name())
							#print("{}-{} is served by {}, expected to finish at {}".format(i,next_actual_act,j, i.get_next_actual_ts()))


	def update_completes(self, completes, ongoing_instance, t):
		for i in ongoing_instance:
			finished = i.check_finished(t)
			if finished==True:
				ongoing_instance.remove(i)
				completes.append(i)
				self.w_comp_time.append(i.get_weighted_comp())
		return completes

	def optimize(self, test_path, json_file_path, model_file_path, mode, **kwargs):
		time1 = time.time()
		t=0
		#initialize
		ongoing_instance = list()
		completes = list()

		self.act_thre=0.8
		self.ts_thre=0.8

		if mode=='test':
			resource_set, instance_set = self.prepare_test(test_path, json_file_path, model_file_path)

		elif mode == 'real':
			if 'org_log_path' in kwargs:
				org_log_path = kwargs['org_log_path']
			else:
				raise AttributeError("no org_log_path given.")
			resource_set, instance_set = self.prepare_real(test_path, org_log_path, json_file_path, model_file_path )

		else:
			raise AttributeError('Optimization mode should be given.')


		while len(instance_set) != len(completes):
			print("{} begins".format(t))
			#ongoing instance를 추가
			ongoing_instance = self.update_ongoing_instances(instance_set, ongoing_instance, t)
			print('current ongoing instance: {}'.format(len(ongoing_instance)))
			G = self.update_object(ongoing_instance, resource_set,t)
			#print('current cand instance and resource: {}, {}'.format(cand_instance, cand_resource))
			M = self.update_plan(G,t)
			#print('current matching: {}'.format(M))
			self.execute_plan(ongoing_instance, resource_set, M, completes, t)
			completes = self.update_completes(completes, ongoing_instance, t)
			print('current completes: {}'.format(len(completes)))
			t+=1
		time2 = time.time()

		print("total weighted sum: {}".format(sum(self.w_comp_time)))
		print('baseline algorithm took {:.3f} ms'.format((time2-time1)*1000.0))