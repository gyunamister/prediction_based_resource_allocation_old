from optimizer.suggested import SuggestedOptimizer

if __name__=='__main__':
	Opt = SuggestedOptimizer()
	"""Experiment on an artificial event log"""
	#To experiment different number of instances, modify 'test_path' (e.g., ./result/testlog_0121_1_40.csv, ./result/testlog_0121_1_60.csv, ...)
	#Prediction model is saved in a directory "./prediction/model" as a combination of json_file and h5_file.
	Opt.optimize(test_path='./sample_data/artificial/testlog_0121_1_40.csv', json_file_path = "./prediction/model/test_model_0121_1.json", model_file_path = "./prediction/model/test_model_0121_1.h5", mode='test')


	"""Experiment on an real-life event log"""
	#Prediction model is saved in a directory "./prediction/model" as a combination of json_file and h5_file.
	Opt.optimize(json_file_path="./prediction/model/BPI_2012_model.json", model_file_path="./prediction/model/BPI_2012_model.h5", org_log_path = './sample_data/real/modi_BPI_2012.csv', test_path = './sample_data/real/modi_BPI_2012_0301.csv', mode='real')

