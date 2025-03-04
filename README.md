# TCT_data
<p align="left"><b>A large annotated cervical cytology images dataset for AI models to aid cervical cancer screening*scientific data*</b></p>
[Journal Link](https://www.nature.com/articles/s41597-025-04374-5)
<p align="left"><img width="600" src="https://github.com/zx333445/TCT_data/blob/main/flow.png?raw=true"></p>

## Installation
Once you clone the repo, please run the following command to create the conda environment.

```bash
$ conda env create --file environment.yaml
```

## Usage

Directory description:

```
├─ network             // directory of detection networks
├─ netdetr             // directory of detr network
├─ netsparse           // directory of sparse rcnn network
├─ netyolo             // directory of yolo network
├─ tool                // directory of tool codes

├─ datasets.py         // dataset code
├─ train.py            // main code for model training
├─ trainer.py          // code for training utils
├─ launch.sh           // train.py launcher
├─ _utils.py           // other utils code
```

Run the following code for model training:

```bash
$ bash launch.sh
```

This will initiate the script `train.py` for 5-fold cross-validation model training. Note that the csv file paths need to be changed according to the actual situation. 

