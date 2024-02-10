# README

This folder contains the source file and numeric records under synthesis settings. The details of setting can be found in the appendix C.1 and C.2 in our paper.

**If you want to rerun our code, please modify the import commands in the notebooks to make sure you can import the source files in the folder "Source."**

## File Structure

+ "L_1": The notebook for the experiments and numeric records, in the case of $L=1$.
+ "L_2": The notebook for the experiments and numeric records, in the case of $L=2$.
+ "Source": The source file of BAI agents
  + agent.py: The implementations of BAI agents.
  + env.py: The implementations of environments that can interact with BAI agents.
  + utils.py, utils_trap.py: The setting of numeric experiments and codes for conducting numeric experiments.
+ K-256_C-1500.eps and K-256_C-1500-1500.eps: Corresponds to the figure 2 in our paper.
+ Visualize-Demo.ipynb: Use the results in folder "L_1" and "L_2" to paint the figures.

