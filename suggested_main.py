from suggested import SuggestedOptimizer
from baseline import BaseOptimizer

if __name__=='__main__':
	#Opt = SuggestedOptimizer()
	#Opt = BaseOptimizer()
	#Opt.optimize(test_path='./result/testlog_0121_1_40.csv', json_file_path = "./prediction/model/test_model_0121_1.json", model_file_path = "./prediction/model/test_model_0121_1.h5", mode='test')


	#real
	Opt = SuggestedOptimizer()
	#Opt = BaseOptimizer()
	Opt.optimize(json_file_path="./prediction/model/BPI_2012_model.json", model_file_path="./prediction/model/BPI_2012_model.h5", org_log_path = './result/modi_BPI_2012_dropna_filter_act.csv', test_path = './result/modi_BPI_2012_0301.csv', mode='real')

