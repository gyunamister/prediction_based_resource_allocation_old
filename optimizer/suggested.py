import sys
import os
from pathlib import Path
import networkx as nx
import time
import numpy as np

p = Path(__file__).resolve().parents[2]
sys.path.append(os.path.abspath(str(p)))

from PyProM.src.data.Eventlog import Eventlog
from object.object import Instance, Resource

def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print('{:s} function took {:.3f} ms'.format(f.__name__, (time2-time1)*1000.0))

        return ret
    return wrap

class SuggestedOptimizer(object):
	def __init__(self, *args, **kwargs):
		super(SuggestedOptimizer, self).__init__(*args, **kwargs)
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

	def initialize_test_resource(self, eventlog):
		resource_set = list()
		resource_list = sorted(list(eventlog.get_resources()))
		for res in resource_list:
			act_list = list(eventlog.loc[eventlog['Resource']==res,'Activity'].unique())
			resource = Resource(res, act_list)
			resource_set.append(resource)
		return resource_set

	def initialize_real_resource(self, test_log):
		resource_set = list()
		resource_list = sorted(list(test_log.get_resources()))
		#resource_list = [str(x) for x in resource_list]

		for res in resource_list:
			act_list = list(test_log.loc[test_log['Resource']==res,'Activity'].unique())
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
		test_log = self.load_data(path=test_path)
		#eventlog['weight'] = eventlog['weight'] * 10
		#resource set 생성 - 어떤 resource가 있고 어떤 acitivity 수행 가능한지 (phase 1에서 dict 형태로 생성)
		resource_set = self.initialize_test_resource(test_log)


		#inital instance 생성
		#set_release_schedule를 참고하여 생성하면 될 듯
		#즉, release_schedule return
		instance_set = self.initialize_test_instance(test_log)
		#prediction을 위한 X 생성을 위해 Instance의 attribute를 생성
		self.set_basic_info(test_log)
		return resource_set, instance_set

	def prepare_real(self, test_path, org_log_path, json_file_path, model_file_path):
		pred_model = self.set_model(json_file_path=json_file_path, model_file_path=model_file_path)
		Instance.set_model(pred_model)
		eventlog = self.load_real_data(path=org_log_path)

		test_log = self.load_real_data(path=test_path)
		test_log['weight'] = test_log['weight']

		instance_set = self.initialize_real_instance(test_log)
		resource_set = self.initialize_real_resource(test_log)

		self.set_basic_info(eventlog)

		return resource_set, instance_set

	#@timing
	def update_ongoing_instances(self, instance_set, ongoing_instance, t):
		for i in instance_set:
			if i.get_release_time() == t:
				ongoing_instance.append(i)
		return ongoing_instance

	#@timing
	def update_object(self, ongoing_instance, resource_set, G, t):
		G = nx.DiGraph()

		for i in ongoing_instance:
			if i.get_next_actual_ts() <= t:
				i.set_status(True)

		for j in resource_set:
			if j.get_next_actual_ts() <= t:
				j.set_status(True)

		for i in ongoing_instance:
			#if already
			if i.get_next_actual_ts() == t:
				i.clear_pred_act_dur()
				next_act = i.get_next_actual_act()
				for j in resource_set:
					if next_act in j.get_skills():
						next_next_pred_act, next_next_act_conf, next_pred_dur = i.predict(next_act, j.get_name())
						i.set_next_act_conf(1)
						i.set_pred_act_dur(j, next_pred_dur)

			elif i.get_next_actual_ts() > t:
				if i.get_next_act_conf() < self.act_thre:
					continue

			for j in i.get_pred_act_dur_dict().keys():
				G.add_edge('s',i, capacity=1)
				G.add_edge(j,'t',capacity=1)
				weight = i.get_weight()
				#j.set_duration_dict(i,pred_dur)
				pred_dur = i.get_pred_act_dur(j)
				pred_dur += max([i.get_next_pred_ts()-t, j.get_next_pred_ts()-t, 0])
				cost = int(pred_dur / weight * 10)
				G.add_edge(i,j,weight=cost,capacity=1, pred_dur=pred_dur)

			"""
			for j in resource_set:
				if next_act in j.get_skills():
					G.add_edge('s',i, capacity=1)
					G.add_edge(j,'t',capacity=1)
					weight = i.get_weight()
					#j.set_duration_dict(i,pred_dur)
					pred_dur = i.get_pred_act_dur(j)
					pred_dur += max([i.get_next_pred_ts()-t, j.get_next_pred_ts()-t, 0])
					cost = int(pred_dur / weight * 10)
					G.add_edge(i,j,weight=cost,capacity=1, pred_dur=pred_dur)
			"""

		return G

	#@timing
	def update_plan(self, G,t):
		#if 's' in nodes and 't' in nodes:
		nodes=G.nodes()
		if len(nodes)!=0:
			M = nx.max_flow_min_cost(G, 's', 't')
			#print(M)
		else:
			M=False
		#print(M)
		#M = MinCost_MaxFlow(s,t) # dict of dict form
		return M

	def modify_plan(self, G, M, t):
		if M!=False:
			for i, _ in M.items():
				if isinstance(i, Instance)==False:
					continue

				temp_dict = dict()
				for j, val in M[i].items():
					if val==1:
						remaining = i.get_next_actual_ts()-t
						#print(remaining)
						if remaining <= 0:
							break
						in_edges_to_j = G.in_edges([j], data=True)
						for source, dest, data in in_edges_to_j:
							if source.get_status()==True:
								if data['pred_dur'] <= remaining:
									temp_dict[source] = source.get_weight()
				#print(temp_dict)
				if len(temp_dict)!=0:
					new_instance = max(temp_dict, key=temp_dict.get)
					M[i][j] = 0
					M[new_instance][j] = 1
					#print("Match changed: from {} to {}".format(i,new_instance))
		return M


	#@timing
	def execute_plan(self, ongoing_instance, resource_set, M, completes, t):
		ready_instance = [x for x in ongoing_instance if x.get_status()==True]
		ready_resource = [x for x in resource_set if x.get_status()==True]

		if M!=False:
			for i in M:
				if isinstance(i, Instance)==False:
					continue
				if i in ready_instance:
					for j, val in M[i].items():
						if val==1:
							if j in ready_resource:
								cur_actual_act = i.get_next_actual_act()
								assigned_res = j.get_name()
								#first prediction
								next_pred_act, next_act_conf, cur_pred_dur = i.predict(cur_actual_act, assigned_res)
								i.set_next_pred_act(next_pred_act)
								i.set_next_act_conf(next_act_conf)
								i.set_next_pred_ts(t+cur_pred_dur)

								#time prediction 후 i 실제 업데이트
								i.update_index()
								i.set_next_actual_act()
								#우선 acutal dur을 pred_dur로 보기 때문에 pred_ts를 next_ts로 설정
								i.set_next_actual_ts(i.get_next_pred_ts())
								i.update_res_history(assigned_res)
								i.set_status(False)

								#time prediction 후 j 실제 업데이트
								j.set_next_pred_ts(t+cur_pred_dur)
								#우선 acutal dur을 pred_dur로 보기 때문에 pred_ts를 next_ts로 설정
								j.set_next_actual_ts(j.get_next_pred_ts())
								j.set_status(False)

								i.clear_pred_act_dur()
								#second prediction
								for k in resource_set:
									if next_pred_act in k.get_skills():
										next_next_pred_act, next_next_act_conf, next_pred_dur = i.predict(next_pred_act, k.get_name())
										i.set_pred_act_dur(k, next_pred_dur)
								#i.set_actual_act_dur()

	#@timing
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

		self.act_thre=0.9
		self.ts_thre=0.8
		self.next_thre = 100

		if mode=='test':
			resource_set, instance_set = self.prepare_test(test_path, json_file_path, model_file_path)

		elif mode == 'real':
			if 'org_log_path' in kwargs:
				org_log_path = kwargs['org_log_path']
			else:
				raise AttributeError("no org_log_path given.")
			resource_set, instance_set = self.prepare_real(test_path, org_log_path, json_file_path, model_file_path )
			print("num resource:{}".format(len(resource_set)))

		else:
			raise AttributeError('Optimization mode should be given.')

		G = nx.DiGraph()

		while len(instance_set) != len(completes):
			print("{} begins".format(t))
			#ongoing instance를 추가
			ongoing_instance = self.update_ongoing_instances(instance_set, ongoing_instance, t)
			print('current ongoing instance: {}'.format(len(ongoing_instance)))
			G = self.update_object(ongoing_instance, resource_set, G,t)
			#print('current cand instance and resource: {}, {}'.format(cand_instance, cand_resource))
			M = self.update_plan(G,t)
			M = self.modify_plan(G, M,t)
			#print('current matching: {}'.format(M))
			self.execute_plan(ongoing_instance, resource_set, M, completes, t)
			completes = self.update_completes(completes, ongoing_instance, t)
			print('current completes: {}'.format(len(completes)))
			t+=1
		time2 = time.time()

		print("total weighted sum: {}".format(sum(self.w_comp_time)))
		print('suggested algorithm took {:.3f} ms'.format((time2-time1)*1000.0))