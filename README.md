

## Summary

Simulation code to explore MSPNs as private representations of data

This repository includes the code used for the paper "Generating synthetic mixed discrete-continuous clinical data with mixed sum-product networks" by Kroes, van Leeuwen, Groenwold and Janssen. In the paper we evaluate whether data generated from MSPNs can be used as an anonymized substitute for original data in various regression analysis scenarios. We measure whether the regression parameter for a variable of interest is measured accurately and precisely and we use interpretable measures of privacy to evaluate whether the data generated by the MSPN can be considered anonymized. 

The file all_functions.py includes all the functions used for the simulation and can be divided into 1. generating data, 2. evaluating privacy, 3. evaluating information loss and 4. printing results. 

The file test_simulation.py includes code to test the simulation code on your own device.

The file results.py includes the code that was run to test the simulations, where we make a distinction between the code needed for table 1 and 2 in the final paper.

The file compute_evars.py contains the code used to compute the error variance needed for R2 of 0.3.

The file anonymize_data_with_mspn contains the function of the same name, which needs the original data and the desired number of clusters as input and it outputs the synthetic data and the MSPN. For data with 100,000 records and 9 variables, the process of generating data can take approximately an hour. 



For any questions, please send an email to shannonkroes@hotmail.com

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



