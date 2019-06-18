# REC_vs_DEEP_and_CONVNN_on_TEMPERATURE_SERIES

Analysis of a weather timeseries dataset using recurrent, deep and convolutional neural networks for temperature forecasting.


### Introduction
In this repository we seek to apply a simple fully connected network, a convnet and different recurrent neural networks to analyze a weather timeseries dataset (the so-called "Jena climate dataset") and predict the air temperature `24` hours in the future.

### Getting started
Computations were done using python 3.7. The needed packages are all listed in the head of the attached Jupiter notebook. In case one of them is not preinstalled, type

```
import sys
!{sys.executable} -m pip install package_name

```
into a new code cell of the notebook.

The dataset itself, can be obtained from the following link: [Jena climate dataset](https://s3.amazonaws.com/keras-datasets/jena_climate_2009_2016.csv.zip).

### Instructions
The project can be run by starting the Jupiter notebook timeseries_analysis.ipynb. This notebook encapsulates all of the different network architectures with respective results and a comparison. A discussion of the analysis is put together in the report.md file.

### License
This repository is licensed under the MIT License.
