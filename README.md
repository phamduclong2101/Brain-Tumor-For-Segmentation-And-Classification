# Brain-Tumor-For-Segmentation-And-Classification
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

The dataset used for this project is taken from Kaggle. You can download it from [Kaggle Brain MRI Dataset](https://www.kaggle.com/). After downloading, make sure to place the dataset in the appropriate directory and update the paths in the `main.py` file accordingly.

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

## Contributing
If you would like to contribute, please open a pull request or issue.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
