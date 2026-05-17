# Goal: Classify facial emotions by category
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight


from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import sys

# Logger setting
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Console will show everything

# Handler to console
stream_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


class ConvolutionalNeuralNetwork:
    def __init__(self):
        self.train_dir = 'C:\\Users\\Weslei\\Desktop\\Assuntos_de_estudo\\Assuntos_de_estudo\\Fases da vida\\Fase I\\Repository Projects\\files\\deep_learning\\train_cnn'
        self.test_dir = 'C:\\Users\\Weslei\\Desktop\\Assuntos_de_estudo\\Assuntos_de_estudo\\Fases da vida\\Fase I\\Repository Projects\\files\\deep_learning\\test_cnn'

        # Image Settings of FER-2013 and batch size for my config
        self.batch_size = 64
        self.img_size = (48, 48)

        self.train_dataset, self.test_dataset = [None] * 2

        self.model_convolutional = None

    def getting_data(self):
        logger.info('Getting train and test data in files')

        self.train_dataset = tf.keras.utils.image_dataset_from_directory(
            self.train_dir,
            labels='inferred',  # Deduce the names of the classes, because we have many categories
            label_mode='categorical',  # Used for multiclass classification
            color_mode='grayscale',  # The FER-2013 is in shades of gray (1 channel).
            batch_size=self.batch_size,
            image_size=self.img_size,
            shuffle=True
        )

        self.test_dataset = tf.keras.utils.image_dataset_from_directory(
            self.test_dir,
            labels='inferred',
            label_mode='categorical',
            color_mode='grayscale',
            batch_size=self.batch_size,
            image_size=self.img_size,
            shuffle=False  # Do not scramble the test to properly evaluate metrics
        )

    @staticmethod
    def model_cnn(learning_rate=0.001):

        model = models.Sequential([
            layers.Rescaling(1. / 255, input_shape=(48, 48, 1)),  # Normalize data [0, 255] to [0, 1]

            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),  # Spatial Invariance
            layers.MaxPooling2D((2, 2)),  # Size 48x48 to 24x24

            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),  # Reduce size again

            layers.Flatten(),  # Adjust matrix in a linear vector
            layers.Dense(128, activation='relu'),  # Deep layers
            layers.Dense(7, activation='softmax')  # Softmax because have 7 class of emotions
        ])

        model.summary()

        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train_and_fit_model(self):
        logger.info('Fit and create a model')

        model = self.model_cnn()

        es = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )

        # It halves the learning rate (factor=0.5) if the loss does not decrease for 2 epochs.
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=0.00001,
            verbose=1
        )

        # Labels of classes
        y_train_labels = np.concatenate([y for x, y in self.train_dataset], axis=0).argmax(axis=1)
        unique_class = np.unique(y_train_labels)
        weight_class = compute_class_weight('balanced', classes=unique_class, y=y_train_labels)
        dict_weight_class = dict(zip(unique_class, weight_class))

        history = model.fit(
            self.train_dataset,
            validation_data=self.test_dataset,
            epochs=30,  # Early stop will work before complete 30
            class_weight=dict_weight_class,
            callbacks=[es, reduce_lr]
        )

        print(history)





    def evaluate_model(self):
        logger.info('Evaluating the best model on test data')
        y_predict = self.best_model.predict(self.X_test)
        y_probs = self.best_model.predict_proba(self.X_test)[:, 1]

        cm = confusion_matrix(self.y_test, y_predict)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.show()

        logger.info("\n--- Classification Report ---")
        logger.info(f'{classification_report(self.y_test, y_predict)}')

        # 4. AUC Score
        auc = roc_auc_score(self.y_test, y_probs)
        logger.info(f"AUC Score: {auc:.4f}")

    def single_image_test(self):
        # Matplotlib para mostrar a foto e o resultado
        pass


class_cnn = ConvolutionalNeuralNetwork()
class_cnn.getting_data()
class_cnn.train_and_fit_model()