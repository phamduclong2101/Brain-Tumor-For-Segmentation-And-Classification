import pandas as pd
from data_generator import DataGenerator
from brain_mri_classifier import BrainMRIClassifier
from brain_mri_segmentation import BrainMRISegmentation

# Load data
dfmask = pd.read_csv('path_to_dfmask.csv')  # Replace with actual path to dfmask

# Initialize and run the classifier
classifier = BrainMRIClassifier(dfmask)
classifier.preprocess_data()
classifier.create_generators()
classifier.build_model()
classifier.train_model()
classifier.load_model()

# Initialize and run the segmentation model
segmentation = BrainMRISegmentation(dfmask)
segmentation.preprocess_data()
segmentation.create_generators()
segmentation.build_model()
segmentation.compile_model()
segmentation.train_model()
segmentation.load_model()

# Make predictions using the segmentation model
test_generator = DataGenerator(segmentation.val_ids, segmentation.val_mask)  # Replace with actual test data
test_predict = segmentation.predict(test_generator)

# Visualize predictions
df_pred = pd.DataFrame({'image_path': segmentation.val_ids, 'predicted_mask': test_predict, 'has_mask': segmentation.val_mask})
segmentation.visualize_predictions(df_pred)
