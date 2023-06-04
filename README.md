# MSPN_privacy


## Summary

This repository includes the code used for the paper "Generating synthetic mixed discrete-continuous health records with mixed sum-product networks" by Kroes, van Leeuwen, Groenwold and Janssen (2022). In the paper we evaluate the performance of cluster-based synthetic data generation. Groups of similar individuals are clustered together and within clusters, the relations between variables are removed, with the aim of protecting privacy. With the collection of clusters (some combinations of values are more likely than other) we aim to retain the relations between the variables. The cluster-based synthetic data generator uses a specific form of a mixed sum-product network (MSPN). In the paper we present the results on whether the synthetic data protect privacy and have the same relations that are found in the original data. The code in this repository can be used to reproduce those results. 

The folder "source" contains all the functions required to run the simulations and to anonymize data. In particular, the file anonymize_data.py is created for external users who want to apply the anonymization approach to their own data, including a small example. The results folder contains all the result objects and the code to analyze these objects. The folder "scripts" contains the code with which these results were generated, using the functions from "source". 

For any questions please send an email to shannon.kroes@tno.nl

## Installation
("myenv" should be replaced by the repository name of your choosing)

```
# Clone the repository
git clone git@github.com:ShannonKroes/MSPN_privacy.git
cd MSPN_privacy

# Create a local conda virtual environment
conda create -n myenv python=3.8
conda activate myenv

# Install the IDE fo your choosing, e.g.
conda install spyder

# Install the required packages
pip install spflow
```
And an important final step: add the altered spn package. First find the spn folder that was installed during installation of spflow (e.g. at envs/myenv/Lib/site-packages or at envs/myenv/lib/python3.8/site-packages) and delete this folder. Then copy the folder MSPN_privacy/source/spn to envs/myenv/Lib/site-packages. 
You're all set! Please let us know if you have any questions.


## Example
A data set can easily be generated with the Simulation class as follows:

```python
from simulation import Simulation
sim = Simulation()
data = sim.generate_data()
```
Then we can anonymize these data with the anonymize_data function:

```python
from source.anonymize_data import anonymize_data
synthetic_data = anonymize_data(data)
```
Note that your own data can also be anonymized, if it is formatted as a numpy array. For example:

```python
import pandas as pd
import numpy as np
my_data = pd.read_csv("my_data.csv").to_numpy()
discrete =  np.array([0,1])
synthetic_data = anonymize_data(my_data, discrete = discrete)
```
Note that we assume that all uniquely identifying information has been removed from "my_data", such as names, adresses, etc and that the 2D array consists of n rows (one for every individual) that all consists of d elements (one for every variable), i.e. the shape is (n,d). In this example we have indicated that the first two variables are discrete-valued and the rest is continuous. Please see the documentation in anonymize_data.py for more details about how to specify which variables should be considered discrete or continuous. Note that these are the only two inputs that are needed to synthesize the data: your data as a numpy array and a numpy array indicating the indices of variables that are discrete (if you have only continuous values you only have to add the data). 

Link to paper: https://pubmed.ncbi.nlm.nih.gov/36228120/

