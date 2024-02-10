# README

This folder contains notebook for real-world numeric experiments and numeric records.

**If you want to run the notebooks in this folder, please change the path of each notebook to make sure they can import the files in the "Source" subfolder.**

## File Structure

Each folder corresponds to a dataset and its numeric experiments. The details of the dataset can be found in the appendix C.3.

+ "Source": Source files for numeric experiments
  + agent.py: Implementation of BAI agents. Our proposed SHRR algorithm corresponds to the class SequentialHalvingRR_Recycle_FailureFlag_History_Agent. We also implemented a SHRR without remaining history, corresponding to the class SequentialHalvingRR_Recycle_FailureFlag_Agent.
  + env.py: Pack the machine learning problem as a environment that will interact with BAI agents.
  + utils.py: Codes for conducting the numeric experiments.
+ "Dataset": This folder contains the data we used to test the machine learning model.
+ "Arcene": This folder contains the notebooks for conducting numeric experiments on dataset Arcene.
+ "Madelon": This folder contains the notebooks for conducting numeric experiments on dataset Madelon.
+ "Mnist38": This folder contains the notebooks for conducting numeric experiments on dataset Mnist 3 and 8.
+ "Obesity": This folder contains the notebooks for conducting numeric experiments on dataset Obesity.
+ "UCIDigits": This folder contains the notebooks for conducting numeric experiments on dataset UCIDigits.

Notebooks for identifying the best machine learning model

+ Experiment-RealWorld-K-32-Find-Best-Arm-arcene.ipynb.
+ Experiment-Realworld-K-32-Find-Best-Arm-digits-3-8.ipynb.
+ Experiment-RealWorld-K-32-Find-Best-Arm-madelon.ipynb.
+ Experiment-RealWorld-K-32-Find-Best-Arm-obesity.ipynb.
+ Experiment-RealWorld-K-32-Find-Best-Arm-UCIdigit.ipynb.