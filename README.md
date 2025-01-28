# Predicting the 3-Dimensional Dose Distribution of Multilesion Lung Stereotactic Ablative Radiation Therapy With Generative Adversarial Networks

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Usage](#usage)
- [Citation](#citation)


## Introduction

This is the companion repository for the paper "Predicting the 3-Dimensional Dose Distribution of Multilesion Lung Stereotactic Ablative Radiation Therapy With Generative Adversarial Networks", published in the International Journal of Radiation Oncology, Biology, Physics.

## Requirements

* Python 3.9
* PyTorch 2.0.1

## Usage
Patient plans should be stored as numpy files, with channels for the CT scan, the initial dose estimation, and contours of the PTVs and OARs. The numpy files should be stored in the directory refenced as DATA_DIR in confV2.yml. Make sure to adjust the file paths for the other directories as well. Use the V2 versions of the code for model training. 


## Citation

If you use this repostiory in your work, please cite: 

```
Wang, E., Abdallah, H., Snir, J., Chong, J., Palma, D. A., Mattonen, S. A., & Lang, P. (2025). Predicting the 3-Dimensional Dose Distribution of Multilesion Lung Stereotactic Ablative Radiation Therapy With Generative Adversarial Networks. International Journal of Radiation Oncology* Biology* Physics, 121(1), 250-260.
```


