The source folder contains:

1. pre train NN data: stores tabular data generated for NN training
2. post train NN data: contains weights, scalars, jupyter notebooks used for training
3. source: code to formulate the NN embeddable MILP, proposed CCG, CCG and exact methods
4. source//instance-generation.jl : Generates instances and uncertainty for the Knapsack Instances = 10, 20 and 30
5. source//Knapsackro_data-generation.ipynb generates the pre training data for NN
6. ARO algorithm solves: plots, results	and codefiles for ML-Accelerated CCG, Accelerated CCG and CCG.