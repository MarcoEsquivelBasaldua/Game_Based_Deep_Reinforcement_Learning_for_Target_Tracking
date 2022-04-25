# Data Set Generation and Motion Policies Approximation in the Continuous State-Space with Artificial Neural Networks

1. To get the train dataset, run:
	- compute_dataset_on_4connectivity_autonomousEvader.py Env\<env>_res\<res>.txt
2. Run the evader_model_classification.ipynb notebook to train the evader model and get:
	- evader_NN_Env\<env>_res\<res>.pt
	- evader_dict_NN_Env\<env>_res\<res>.pt
3. Run the pursuer_model_classification.ipynb notebook to train the pursuer model and get:
	- pursuer_NN_Env\<env>_res\<res>.pt
	- pursuer_dict_NN_Env\<env>_res\<res>.pt
4. To simulate a game using the trained models, run:
	- policies_approximation.py Env\<env>_res\<res>.txt
5. Run the results_stats_complete.ipynb notebook to get the results comparing every model performance.
