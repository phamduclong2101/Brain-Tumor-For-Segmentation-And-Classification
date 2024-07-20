# Brain MRI Classification and Segmentation

This repository contains code for Brain MRI Classification and Segmentation using deep learning models.

## Directory Structure


├── data_generator.py 

├── brain_mri_classifier.py

├── brain_mri_segmentation.py

├── main.py

├── README.md

├── requirements.txt

└── data_mask.csv

## Files Description

- `data_generator.py`: Contains the `DataGenerator` class for generating batches of data for training and validation.
- `brain_mri_classifier.py`: Contains the `BrainMRIClassifier` class for training and predicting using a classification model.
- `brain_mri_segmentation.py`: Contains the `BrainMRISegmentation` class for training and predicting using a segmentation model.
- `main.py`: Main file to run the classification and segmentation models.
- `requirements.txt`: List of required Python packages.
- `data_mask.csv`: CSV file containing paths to images and their corresponding masks.

## Dataset

The dataset used for this project is taken from Kaggle. You can download it from [Kaggle Brain MRI Dataset]([https://www.kaggle.com/](https://www.kaggle.com/datasets/arcticai/brain-mri-detection-and-segmentation)). After downloading, make sure to place the dataset in the appropriate directory and update the paths in the `main.py` file accordingly.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/brain-mri-classification-segmentation.git
cd brain-mri-classification-segmentation
```

2. Create a virtual environment and activate it:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. Install the required packages:
   
```bash
pip install -r requirements.txt
```

## Usage

1. Prepare your data: Download the dataset from Kaggle and place it in the correct directory. Update the paths in the main.py file accordingly.
2. Run the main script:
   
```bash
python main.py
```

## Customizing
Modify the parameters in data_generator.py to customize the data generation process.
Adjust the model architecture and training parameters in brain_mri_classifier.py and brain_mri_segmentation.py as needed.

------------------------------------------
## Training results

### Training graphs
### classification graphs
<br> <img width="500" alt="RGB" src="https://github.com/phamduclong2101/Brain-Tumor-For-Segmentation-And-Classification/blob/992f63fe488ddb4e2b15f310eddf82efbba27d14/Results/a.jpg">
### classification graphs
<br> <img width="500" alt="RGB" src="https://github.com/phamduclong2101/Brain-Tumor-For-Segmentation-And-Classification/blob/992f63fe488ddb4e2b15f310eddf82efbba27d14/Results/b.jpg">

### Sample outputs
<br> <img width="500" alt="RGB" src="https://github.com/phamduclong2101/Brain-Tumor-For-Segmentation-And-Classification/blob/992f63fe488ddb4e2b15f310eddf82efbba27d14/Results/brain_mri_samples1.png">

## Contributing
If you would like to contribute, please open a pull request or issue.


### **Technologies used**
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)


### **Tools used**
![PyCharm](https://img.shields.io/badge/pycharm-143?style=for-the-badge&logo=pycharm&logoColor=black&color=black&labelColor=green)
![Git](https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white)
![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)

<!-- Tools Used -->
[PyCharm]: https://code.visualstudio.com/
[git]: https://git-scm.com/
[github]: https://github.com/
[python]: https://www.python.org/
[sklearn]: https://scikit-learn.org/stable/

### Authors

Pham Duc Long

## To Read:
1. [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597.pdf)
2. [Resnet: Deep Residual Learning for Image Recognition](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)
