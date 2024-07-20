import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, MaxPool2D, UpSampling2D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from data_generator import DataGenerator

def tversky(y_true, y_pred, smooth=1e-6):
    y_true_pos = tf.keras.backend.flatten(y_true)
    y_pred_pos = tf.keras.backend.flatten(y_pred)
    true_pos = tf.keras.backend.sum(y_true_pos * y_pred_pos)
    false_neg = tf.keras.backend.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = tf.keras.backend.sum((1 - y_true_pos) * y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)

def focal_tversky(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return tf.keras.backend.pow((1 - pt_1), gamma)

class BrainMRISegmentation:
    def __init__(self, dfmask, base_path='./'):
        self.dfmask = dfmask
        self.base_path = base_path
        self.model = None

    def preprocess_data(self):
        dftum = self.dfmask[self.dfmask['mask'] == 1]
        X_train, X_val = train_test_split(dftum, test_size=0.15)
        X_test, X_val = train_test_split(X_val, test_size=0.5)
        self.train_ids = list(X_train.image_path)
        self.train_mask = list(X_train.mask_path)
        self.val_ids = list(X_val.image_path)
        self.val_mask = list(X_val.mask_path)

    def create_generators(self):
        self.training_generator = DataGenerator(self.train_ids, self.train_mask)
        self.validation_generator = DataGenerator(self.val_ids, self.val_mask)

    def resblock(self, X, f):
        X_copy = X
        X = Conv2D(f, kernel_size=(1, 1), strides=(1, 1), kernel_initializer='he_normal')(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)
        X = Conv2D(f, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal')(X)
        X = BatchNormalization()(X)
        X_copy = Conv2D(f, kernel_size=(1, 1), strides=(1, 1), kernel_initializer='he_normal')(X_copy)
        X_copy = BatchNormalization()(X_copy)
        X = Add()([X, X_copy])
        X = Activation('relu')(X)
        return X

    def upsample_concat(self, x, skip):
        x = UpSampling2D((2, 2))(x)
        merge = Concatenate()([x, skip])
        return merge

    def build_model(self, input_shape=(256, 256, 3)):
        X_input = Input(input_shape)
        conv1_in = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(X_input)
        conv1_in = BatchNormalization()(conv1_in)
        conv1_in = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1_in)
        conv1_in = BatchNormalization()(conv1_in)
        pool_1 = MaxPool2D(pool_size=(2, 2))(conv1_in)
        conv2_in = self.resblock(pool_1, 32)
        pool_2 = MaxPool2D(pool_size=(2, 2))(conv2_in)
        conv3_in = self.resblock(pool_2, 64)
        pool_3 = MaxPool2D(pool_size=(2, 2))(conv3_in)
        conv4_in = self.resblock(pool_3, 128)
        pool_4 = MaxPool2D(pool_size=(2, 2))(conv4_in)
        conv5_in = self.resblock(pool_4, 256)
        up_1 = self.upsample_concat(conv5_in, conv4_in)
        up_1 = self.resblock(up_1, 128)
        up_2 = self.upsample_concat(up_1, conv3_in)
        up_2 = self.resblock(up_2, 64)
        up_3 = self.upsample_concat(up_2, conv2_in)
        up_3 = self.resblock(up_3, 32)
        up_4 = self.upsample_concat(up_3, conv1_in)
        up_4 = self.resblock(up_4, 16)
        output = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(up_4)
        self.model = Model(inputs=X_input, outputs=output)

    def compile_model(self, learning_rate=0.05, epsilon=0.1):
        adam = Adam(lr=learning_rate, epsilon=epsilon)
        self.model.compile(optimizer=adam, loss=focal_tversky, metrics=[tversky])

    def train_model(self, epochs=50, patience=20):
        earlystopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience)
        checkpointer = ModelCheckpoint(filepath="segmentation-weights.hdf5", verbose=1, save_best_only=True)
        self.model.fit(
            self.training_generator,
            epochs=epochs,
            validation_data=self.validation_generator,
            callbacks=[checkpointer, earlystopping]
        )
        self.model.save_weights("segmentation-weights.hdf5")
        model_json = self.model.to_json()
        with open("segmentation-model.json", "w") as json_file:
            json_file.write(model_json)

    def load_model(self):
        with open('segmentation-model.json', 'r') as json_file:
            json_savedModel = json_file.read()
        self.model = tf.keras.models.model_from_json(json_savedModel)
        self.model.load_weights('segmentation-weights.hdf5')
        self.compile_model()

    def predict(self, test_generator):
        test_predict = self.model.predict(test_generator, steps=test_generator.n // 16, verbose=1)
        return test_predict

    def prediction(self, test, model):
        directory = "./"
        mask = []
        image_id = []
        has_mask = []
        for i in test.image_path:
            path = directory + str(i)
            img = io.imread(path)
            img = img * 1. / 255.
            img = cv2.resize(img, (256, 256))
            img = np.array(img, dtype=np.float64)
            img = np.reshape(img, (1, 256, 256, 3))
            is_defect = model.predict(img)
            if np.argmax(is_defect) == 0:
                image_id.append(i)
                has_mask.append(0)
                mask.append('No mask')
                continue
            img = io.imread(path)
            X = np.empty((1, 256, 256, 3))
            img = cv2.resize(img, (256, 256))
            img = np.array(img, dtype=np.float64)
            img -= img.mean()
            img /= img.std()
            X[0,] = img
            predict = self.model.predict(X)
            if predict.round().astype(int).sum() == 0:
                image_id.append(i)
                has_mask.append(0)
                mask.append('No mask')
            else:
                image_id.append(i)
                has_mask.append(1)
                mask.append(predict)
        return image_id, mask, has_mask

    def visualize_predictions(self, df_pred, count=10):
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(count, 5, figsize=(30, 50))
        count = 0
        for i in range(len(df_pred)):
            if df_pred['has_mask'][i] == 1 and count < 10:
                img = io.imread(df_pred.image_path[i])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                axs[count][0].title.set_text("Brain MRI")
                axs[count][0].imshow(img)
                mask = io.imread(df_pred.mask_path[i])
                axs[count][1].title.set_text("Original Mask")
                axs[count][1].imshow(mask)
                predicted_mask = np.asarray(df_pred.predicted_mask[i])[0].squeeze().round()
                axs[count][2].title.set_text("Predicted Mask")
                axs[count][2].imshow(predicted_mask)
                img[mask == 255] = (255, 0, 0)
                axs[count][3].title.set_text("MRI with original Mask")
                axs[count][3].imshow(img)
                img_ = io.imread(df_pred.image_path[i])
                img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
                img_[predicted_mask == 1] = (0, 255, 0)
                axs[count][4].title.set_text("MRI with predicted Mask")
                axs[count][4].imshow(img_)
                count += 1
        fig.tight_layout()
        plt.show()
