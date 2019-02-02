Following is the supplementary material for the article "Prediction-based Resource Allocation using LSTM and maximum ﬂow and minimum cost algorithm" by Gyunam Park and [Minseok Song](http://mssong.postech.ac.kr) submitted to the [1st International Conference on Process Mining](https://icpmconference.org).

The code provided in this repository can be readily used to optimize resource scheduling in non-clairvoyant online job shop environment

### Requirements:

- This code is written in Python3.6. In addition, you need to install a few more packages.

- To install,

  ```
  $ cd prediction_based_resource_allocation
  $ pip install -r requirements.txt
  ```



### Implementation:

- **For Evaluation**
  - `python baseline_main.py`: This script optimizes resource scheduling with baseline approach, WeightGreedy. Detailed guide is provided in baseline_main.py
  - `python suggested_main.py`: This script optimizes resource scheduling with the suggest method(two-phase method). Details in suggested_main.py
  - results: total weighted completion time and computing time
- **Brief Explanation**
  - We simulate an emergency department of a hospital to generate artificial eventlogs. Detailed implementation in data/log_generator.py
  - We build an event stream from event log. Detailed implementation in data/preprocess.py
  - Phase 1 of our method (prediction model construction) is implemented in prediction/new_lstm.py. The resulting prediction model is saved in a directory "./prediction/model" as a combination of json_file and h5_file.
  - Phase 2 of our method (prediction model construction) is implemented in optimizer/suggested.py, while the baseline method is implemented in optimizer/baseline.py