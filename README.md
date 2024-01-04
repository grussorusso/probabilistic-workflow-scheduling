This repository contains code and data used for the experiments presented
in the following manuscript:

	G. Russo Russo, R. Marotta, F. Cordari, F. Quaglia, V. Cardellini,
	P. Di Sanzo, "Efficient Probabilistic Workflow Scheduling for IaaS Clouds", 2023 (under review)

Please find below instructions to use the code.

## Requirements

Creating a virtual environment (e.g., using `venv`) is recommended.
To install the dependencies:

	pip install -r requirements.txt

## Running

A single execution of any scheduling algorithm can be launched as follows:

	python sched_experiments.py --job <JOB> \
	       --algorithm <ALG> \
	       --deadline <DEADLINE_SECONDS> \
	       --max_vmtypes <N>

Supported jobs are: `epigenomics`, `sipht`, `cybershake`, `montage`, `ligo`.

Available algorithms:

 - **`EPOSS`** (our novel algorithm)
 - **`P-EPOSS`** (parallel version of EPOSS)
 - `Random`
 - `HEFT`
 - `CloudMOHEFT`
 - `GreedyCost`
 - `Genetic`
 - `Dyna`

To see all the available options:

	python sched_experiments.py -h

To run the full comparison of algorithms presented in the manuscript (it takes
several hours):

	python sched_experiments.py --experiment a

The results are saved in `resultsMainComparison.csv`.
