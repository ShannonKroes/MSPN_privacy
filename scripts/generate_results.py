# -*- coding: utf-8 -*-
"""
In this file we generate the privacy and information loss results. It takes a long time to run this code, so it is not adviced on a personal computer.
Please find a test run under scripts.tests.privacy_test.py and the full results can be found in the folder results.privacy. 
"""
import os
from evaluate_privacy_information_loss import simulation_spn_privacy
ans= np.array([100000])


n30= simulation_spn_privacy(distr="normal", repetitions=10000, threshold=-1, mis=999, n=1000, rows="kmeans", no_clusters=40, H0=True, an_sample_size=ans, standardize=True, priv_an_runs= 50, no_tests=1000, priv_or_runs=50)

n40= simulation_spn_privacy(distr="normal", repetitions=500, threshold=-1, mis=9999, n=10000, rows="kmeans", no_clusters=400, H0=True, an_sample_size=ans, standardize=True, priv_an_runs= 50, no_tests=1000, priv_or_runs=50)

n50= simulation_spn_privacy(distr="normal", repetitions=50, threshold=-1, mis=9999, n=100000, rows="kmeans", no_clusters=4000, H0=True, an_sample_size=ans, standardize=True, priv_an_runs= 50, no_tests=1000, priv_or_runs=50)


p30= simulation_spn_privacy(distr="poisson", repetitions=10000, threshold=-1, mis=999, n=1000, rows="kmeans", no_clusters=40, H0=True, an_sample_size=ans, standardize=True, priv_an_runs= 50, no_tests=1000, priv_or_runs=50)

p40= simulation_spn_privacy(distr="poisson", repetitions=500, threshold=-1, mis=9999, n=10000, rows="kmeans", no_clusters=400, H0=True, an_sample_size=ans, standardize=True, priv_an_runs= 50, no_tests=1000, priv_or_runs=50)

p50= simulation_spn_privacy(distr="poisson", repetitions=50, threshold=-1, mis=9999, n=100000, rows="kmeans", no_clusters=4000, H0=True, an_sample_size=ans, standardize=True, priv_an_runs= 50, no_tests=1000, priv_or_runs=50)


c30= simulation_spn_privacy(distr="categorical", repetitions=10000, threshold=-1, mis=999, n=1000, rows="kmeans", no_clusters=40, H0=True, an_sample_size=ans, standardize=True, priv_an_runs= 50, no_tests=1000, priv_or_runs=50)

c40= simulation_spn_privacy(distr="categorical", repetitions=500, threshold=-1, mis=9999, n=10000, rows="kmeans", no_clusters=400, H0=True, an_sample_size=ans, standardize=True, priv_an_runs= 50, no_tests=1000, priv_or_runs=50)

c50= simulation_spn_privacy(distr="categorical", repetitions=50, threshold=-1, mis=9999, n=100000, rows="kmeans", no_clusters=4000, H0=True, an_sample_size=ans, standardize=True, priv_an_runs= 50, no_tests=1000, priv_or_runs=50)


m30= simulation_spn_privacy(distr="mixed", repetitions=10000, threshold=-1, mis=999, n=1000, rows="kmeans", no_clusters=40, H0=True, an_sample_size=ans, standardize=True, priv_an_runs= 50, no_tests=1000, priv_or_runs=50)

m40= simulation_spn_privacy(distr="mixed", repetitions=500, threshold=-1, mis=9999, n=10000, rows="kmeans", no_clusters=400, H0=True, an_sample_size=ans, standardize=True, priv_an_runs= 50, no_tests=1000, priv_or_runs=50)

m50= simulation_spn_privacy(distr="mixed", repetitions=50, threshold=-1, mis=9999, n=100000, rows="kmeans", no_clusters=4000, H0=True, an_sample_size=ans, standardize=True, priv_an_runs= 50, no_tests=1000, priv_or_runs=50)



n31= simulation_spn_privacy(distr="normal", repetitions=10000, threshold=-1, mis=999, n=1000, rows="kmeans", no_clusters=40, H0=False, an_sample_size=ans, standardize=True, priv_an_runs= 50, no_tests=1000, priv_or_runs=50)

n41= simulation_spn_privacy(distr="normal", repetitions=500, threshold=-1, mis=9999, n=10000, rows="kmeans", no_clusters=400, H0=False, an_sample_size=ans, standardize=True, priv_an_runs= 50, no_tests=1000, priv_or_runs=50)

n51= simulation_spn_privacy(distr="normal", repetitions=50, threshold=-1, mis=9999, n=100000, rows="kmeans", no_clusters=4000, H0=False, an_sample_size=ans, standardize=True, priv_an_runs= 50, no_tests=1000, priv_or_runs=50)


p31= simulation_spn_privacy(distr="poisson", repetitions=10000, threshold=-1, mis=999, n=1000, rows="kmeans", no_clusters=40, H0=False, an_sample_size=ans, standardize=True, priv_an_runs= 50, no_tests=1000, priv_or_runs=50)

p41= simulation_spn_privacy(distr="poisson", repetitions=500, threshold=-1, mis=9999, n=10000, rows="kmeans", no_clusters=400, H0=False, an_sample_size=ans, standardize=True, priv_an_runs= 50, no_tests=1000, priv_or_runs=50)

p51= simulation_spn_privacy(distr="poisson", repetitions=50, threshold=-1, mis=9999, n=100000, rows="kmeans", no_clusters=4000, H0=False, an_sample_size=ans, standardize=True, priv_an_runs= 50, no_tests=1000, priv_or_runs=50)


c31= simulation_spn_privacy(distr="categorical", repetitions=10000, threshold=-1, mis=999, n=1000, rows="kmeans", no_clusters=40, H0=False, an_sample_size=ans, standardize=True, priv_an_runs= 50, no_tests=1000, priv_or_runs=50)

c41= simulation_spn_privacy(distr="categorical", repetitions=500, threshold=-1, mis=9999, n=10000, rows="kmeans", no_clusters=400, H0=False, an_sample_size=ans, standardize=True, priv_an_runs= 50, no_tests=1000, priv_or_runs=50)

c51= simulation_spn_privacy(distr="categorical", repetitions=50, threshold=-1, mis=9999, n=100000, rows="kmeans", no_clusters=4000, H0=False, an_sample_size=ans, standardize=True, priv_an_runs= 50, no_tests=1000, priv_or_runs=50)


m31= simulation_spn_privacy(distr="mixed", repetitions=10000, threshold=-1, mis=999, n=1000, rows="kmeans", no_clusters=40, H0=False, an_sample_size=ans, standardize=True, priv_an_runs= 50, no_tests=1000, priv_or_runs=50)

m41= simulation_spn_privacy(distr="mixed", repetitions=500, threshold=-1, mis=9999, n=10000, rows="kmeans", no_clusters=400, H0=False, an_sample_size=ans, standardize=True, priv_an_runs= 50, no_tests=1000, priv_or_runs=50)

m51= simulation_spn_privacy(distr="mixed", repetitions=50, threshold=-1, mis=9999, n=100000, rows="kmeans", no_clusters=4000, H0=False, an_sample_size=ans, standardize=True, priv_an_runs= 50, no_tests=1000, priv_or_runs=50)
