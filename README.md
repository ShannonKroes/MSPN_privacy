# MSPN_privacy


## Summary

Simulation code to explore MSPNs as private representations of data

This repository includes the code used for the paper "Generating synthetic mixed discrete-continuous health records with mixed sum-product networks" by Kroes, van Leeuwen, Groenwold and Janssen. In the paper we evaluate whether data generated from MSPNs can be used as an anonymized substitute for original data in various regression analysis scenarios. We measure whether the regression parameter for a variable of interest is measured accurately and precisely and we use interpretable measures of privacy to evaluate whether the data generated by the MSPN can be considered anonymized. 

The folder "source" contains all the functions required to run the simulations and to anonymize data. In particular, the file anonymize_data.py is created for external users who want to apply the anonymization approach to their own data, including a small example. The results folder contains all the result objects and the code to analyze these objects. The folder "scripts" contains the code with which these results were generated, using the functions from "source". 

For any questions please send an email to m.janssen@sanquin.nl

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

## Installation

To be able to run the code in this repository, follow the instructions below.
These instructions assume you have a working Python 3 installation.

```
# Clone the repository
git clone git@github.com:ShannonKroes/MSPN_privacy.git
cd MSPN_privacy

# Create a local virtual environment
python -m venv venv

# Activate the virtual environment
source venv/bin/activate      # Unix
# -OR- 
source venv/Scripts/activate  # Windows

# Install required packages
pip install -r requirements.txt
```



