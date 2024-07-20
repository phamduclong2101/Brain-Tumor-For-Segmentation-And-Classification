import numpy as np
from sklearn.model_selection import train_test_split
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, AveragePooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf

class BrainMRIClassifier:
    def __init__(self, dfmask, base_path='./'):
        self.dfmask = dfmask
        self.base_path = base_path
        self.model = None

    def preprocess_data(self):
        traindf = self.dfmask.drop('patient_id', axis=1)
        traindf['mask'] = traindf['mask'].apply(lambda x: str(x))
        self.train, self.test = train_test_split(traindf, test_size=0.15)

    def create_generators(self):
        datagen = ImageDataGenerator(rescale=1./255., validation_split=0.15)
        self.train_generator = datagen.flow_from_dataframe(
            dataframe=self.train,
            directory=self.base_path,
            x_col='image_path',
            y_col='mask',
            subset="training",
            batch_size=16,
            shuffle=True,
            class_mode="categorical",
            target_size=(256, 256)
        )
        self.valid_generator = datagen.flow_from_dataframe(
            dataframe=self.train,
            directory=self.base_path,
            x_col='image_path',
            y_col='mask',
            subset="validation",
            batch_size=16,
            shuffle=True,
            class_mode="categorical",
            target_size=(256, 256)
        )
        test_datagen = ImageDataGenerator(rescale=1./255.)
        self.test_generator = test_datagen.flow_from_dataframe(
            dataframe=self.test,
            directory=self.base_path,
            x_col='image_path',
            y_col='mask',
            batch_size=16,
            shuffle=False,
            class_mode='categorical',
            target_size=(256, 256)
        )

    def build_model(self):
        basemodel = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(256, 256, 3)))
        for layer in basemodel.layers:
            layer.trainable = False
        headmodel = basemodel.output
        headmodel = AveragePooling2D(pool_size=(4, 4))(headmodel)
        headmodel = Flatten(name='flatten')(headmodel)
        headmodel = Dense(256, activation="relu")(headmodel)
        headmodel = Dropout(0.3)(headmodel)
        headmodel = Dense(256, activation="relu")(headmodel)
        headmodel = Dropout(0.3)(headmodel)
        headmodel = Dense(2, activation='softmax')(headmodel)
        self.model = Model(inputs=basemodel.input, outputs=headmodel)
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

    def train_model(self, epochs=50, patience=20):
        earlystopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience)
        checkpointer = ModelCheckpoint(filepath="classifier-resnet-weights.hdf5", verbose=1, save_best_only=True)
        self.model.fit(
            self.train_generator,
            steps_per_epoch=self.train_generator.n // 32,
            epochs=epochs,
            validation_data=self.valid_generator,
            validation_steps=self.valid_generator.n // 32,
            callbacks=[checkpointer, earlystopping]
        )
        self.model.save_weights("classifier-resnet-weights.hdf5")
        model_json = self.model.to_json()
        with open("classifier-resnet-model.json", "w") as json_file:
            json_file.write(model_json)

    def load_model(self):
        with open('classifier-resnet-model.json', 'r') as json_file:
            json_savedModel = json_file.read()
        self.model = tf.keras.models.model_from_json(json_savedModel)
        self.model.load_weights('classifier-resnet-weights.hdf5')
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

    def predict(self, test_generator):
        test_predict = self.model.predict(test_generator, steps=test_generator.n // 16, verbose=1)
        predict = [str(np.argmax(i)) for i in test_predict]
        return np.asarray(predict)
