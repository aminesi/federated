# FL Analysis 

This repository contains the code and results for the paper "Robustness Analysis of Federated Learning Algorithms" submitted to ICSE 2022.

## Replication

### Main experiment
All experiments are done using python 3.8 and TensorFlow 2.4

Steps to run the experiments are as follows:

1. The options for each configuration are set in JSON file which should be in the root directory by default. 
   However, this can be changed using the environment variable CONFIG_PATH.
   
2. The paths for the output and the processed ADNI dataset is set using the environment variables RESULTS_ROOT and ADNI_ROOT respectively.
   If these variables are not set the mentioned paths will use "./results" and "./adni" as default.
   
3. Run the main program by ``` python test.py```

* Note that the results will be overwritten if same config is run for multiple time. To avoid that RESULTS_ROOT can be changed at each run.

### Config details

The config file can have the following options: 

```yaml
    "dataset": one of the following 
      "adni"
      "mnist"
      "cifar"
    "aggregator": one of the following 
      "fed-avg"
      "median"
      "trimmed-mean"
      "krum"
      "combine"
    "attack": one of the following
      "label-flip"
      "noise-data"
      "overlap-data"
      "delete-data"
      "unbalance-data"
      "random-update"
      "sign-flip"
      "backdoor"
    "attack-fraction": a float between 0 and 1
    "non-iid-deg": a float between 0 and 1
    "num-rounds": an integer value
```

Notes:
1. attack field is optional. If it is not present, no attack will be applied and attack-fraction is not necessary.
2. If dataset is set to adni, non-iid-deg field is not necessary
3. The aggregator field is optional and if it is not present it will use the default fed-avg.
4. All configurations used in our experiments are available in `configs` folder

### ADNI dataset

ADNI dataset is not included in the repository due to user agreements, but information about it is available in www.adni-info.org.

Once the dataset is available, data can be processed with `extract_central_axial_slices_adni.ipynb`

### Results Visualization

Results can be visualized using the `visualizer.ipynb`.

* The root folder of the results should be set in the notebook before running.
* Visualizations will be saved in the root folder under 0images folder.
* The visualizer expects the root sub folders to be the results of the different runs.

An example:

```

_root
├── _run1
│   ├── cifar-0--fedavg--clean
│   └── cifar-0--krum--clean
├── _run2
│   ├── cifar-0--fedavg--clean
│   └── cifar-0--krum--clean
└── _run3
    ├── cifar-0--fedavg--clean
    └── cifar-0--krum--clean


```

## Results 

All results are available in the results folder (ADNI, CIFAR, Fashion MNIST, Ensemble).
Each sub folder that represents a dataset contains the details of runs, plus processed visualizations and raw csv files in a folder called 0images.
