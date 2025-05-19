import sys
from PyQt5.QtWidgets import QApplication, QFileDialog, QDialog, QPushButton, QGraphicsView, QGraphicsScene, QLabel, QProgressBar
from PyQt5 import uic
from PyQt5.QtGui import QPixmap, QImage
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PyQt5.QtCore import QTimer
import joblib
from sklearn.linear_model import LogisticRegression


import tensorflow as tf
print(tf.__version__)
# Check if TensorFlow is using GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print("Num GPUs Available: ", len(physical_devices))
else:
    print("No GPU available, using CPU.")



class ImageClassifier(QDialog):
    def __init__(self):
        super(ImageClassifier, self).__init__()
        uic.loadUi(r'G:/data science certificates/GUI project/GUI Detection.ui', self)
        print("UI loaded successfully")
        self.model = None  # Initially no model is loaded

        

        # Load buttons
        self.load_button_1 = self.findChild(QPushButton, 'pushButton')
        self.load_button_1.clicked.connect(self.load_image)

        self.load_button_2 = self.findChild(QPushButton, 'pushButton_2')
        self.load_button_2.clicked.connect(self.predict_with_model)

        # Buttons to load different models
        self.model_button_1 = self.findChild(QPushButton, 'pushButton_3')
        self.model_button_2 = self.findChild(QPushButton, 'pushButton_4')
        self.model_button_3 = self.findChild(QPushButton, 'pushButton_5')
        self.model_button_4 = self.findChild(QPushButton, 'pushButton_6')
        self.model_button_5 = self.findChild(QPushButton, 'pushButton_7')
        self.model_button_6 = self.findChild(QPushButton, 'pushButton_8')
        self.meta_model_button = self.findChild(QPushButton, 'pushButton_meta')
        

        if self.model_button_1:
            self.model_button_1.clicked.connect(lambda: self.load_model_by_name("Model 1"))
        else:
            print("Model button 1 not found!")

        if self.model_button_2:
            self.model_button_2.clicked.connect(lambda: self.load_model_by_name("Model 2"))
        else:
            print("Model button 2 not found!")

        if self.model_button_3:
            self.model_button_3.clicked.connect(lambda: self.load_model_by_name("Model 3"))
        else:
            print("Model button 3 not found!")
        
        if self.model_button_4:
            self.model_button_4.clicked.connect(lambda: self.load_model_by_name("Model 4"))
        else:
            print("Model button 4 not found!")
        
        if self.model_button_5:
            self.model_button_5.clicked.connect(lambda: self.load_model_by_name("Model 5"))
        else:  
            print("Model button 5 not found!")
        
        if self.model_button_6:
            self.model_button_6.clicked.connect(lambda: self.load_model_by_name("Model 6"))
        else:
            print("Model button 6 not found!")

                # New Meta Model Button
    
        if self.meta_model_button:
            self.meta_model_button.clicked.connect(lambda: self.load_model_by_name("Stacked Model"))
        else:
            print("Meta model button not found!")

        # Image viewer
        self.image_viewer_1 = self.findChild(QGraphicsView, 'graphicsView')
        self.scene = QGraphicsScene(self)
        self.image_viewer_1.setScene(self.scene)

        # Result labels
        self.result_label = self.findChild(QLabel, 'label')
        self.result_label_2 = self.findChild(QLabel, 'label_2')

        # Model paths
        self.model_paths = {
        "Model 1": r'C:/Users/osc/Desktop/bestmodel.h5',
        "Model 2": r'C:/Users/osc/Desktop/cnn_model/best_VGG16_model.h5',
        "Model 3": r'C:/Users/osc/Desktop/cnn_model/best_VGG19_model.h5',
        "Model 4": r'C:/Users/osc/Desktop/cnn_model/best_ResNet152V2_model.h5',
        "Model 5": r'C:/Users/osc/Desktop/cnn_model/best_InceptionV3_model.h5',
        "Model 6": r'C:/Users/osc/Desktop/cnn_model/best_DenseNet_model.h5',
        "Stacked Model": r"C:/Users/osc/Desktop/cnn_model/stacked_meta_model.pkl"
    }

        self.progressBar = self.findChild(QProgressBar, 'progressBar')
        self.progressBar.setVisible(False)


        # Class names mapping
        self.classes = {0: 'buildings', 1: 'forest', 2: 'glacier', 3: 'mountain', 4: 'sea', 5: 'street'}

        self.image_path = None



    def update_progress_and_load_stacked(self, model_path):
        if self.progress_value < 100:
            self.progress_value += 5
            self.progressBar.setValue(self.progress_value)
        else:
            self.timer.stop()
            try:
                self.model = joblib.load(model_path)
                self.result_label.setText("Stacked model loaded successfully.")
                print("Stacked model loaded successfully.")
            except Exception as e:
                self.result_label.setText(f"Failed to load stacked model: {e}")
                print(f"Failed to load stacked model: {e}")
            finally:
                self.progressBar.setVisible(False)

    def load_model_by_name(self, model_name):
        try:
            model_path = self.model_paths.get(model_name)
            print(f"Trying to load model from: {model_path}")

            if not model_path:
                self.result_label.setText(f"Model path for {model_name} not found.")
                return

            # Show ProgressBar & reset value
            self.progressBar.setVisible(True)
            self.progressBar.setValue(0)
            self.progress_value = 0

            self.timer = QTimer()

            # Decide which loader to use
            if model_name == "Stacked Model":
                self.timer.timeout.connect(lambda: self.update_progress_and_load_stacked(model_path))
            else:
                self.timer.timeout.connect(lambda: self.update_progress_and_load(model_path, model_name))

            self.timer.start(30)

        except Exception as e:
            self.result_label.setText(f"Failed to load {model_name} from {model_path}: {e}")
            print(f"Failed to load {model_name} from {model_path}: {e}")
            self.result_label_2.setText("")


    def update_progress_and_load(self, model_path, model_name):
        if self.progress_value < 100:
            self.progress_value += 5
            self.progressBar.setValue(self.progress_value)
        else:
            self.timer.stop()
            try:
                self.model = load_model(model_path, compile=False)
                self.result_label.setText(f"{model_name} loaded successfully.")
                print(f"{model_name} loaded successfully.")
            except Exception as e:
                self.result_label.setText(f"Failed to load {model_name}: {e}")
                print(f"Failed to load {model_name}: {e}")
            finally:
                self.progressBar.setVisible(False)

    def load_image(self):
        options = QFileDialog.Options()
        self.image_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.xpm *.jpg *.jpeg);;All Files (*)", options=options)
        if self.image_path:
            try:
                # Load and resize image for viewing
                img = cv2.imread(self.image_path)
                img_resized = cv2.resize(img, (224, 224))  # Resize for display
                height, width, _ = img_resized.shape
                bytes_per_line = 3 * width
                q_img = QImage(img_resized.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()

                self.scene.clear()
                self.scene.addPixmap(QPixmap.fromImage(q_img))
                self.image_viewer_1.setScene(self.scene)
            except Exception as e:
                self.result_label.setText(f"Error loading image: {e}")

    def preprocess_image(self, image_path):
        try:
            img = cv2.imread(image_path)
            img = cv2.resize(img, (100, 100))  # Resize to model input size (make sure it matches model input)
            img = img.astype('float32') / 255.0  # Normalize
            img = np.expand_dims(img, axis=0)  # Add batch dimension
            return img
        except Exception as e:
            self.result_label.setText(f"Error processing image: {e}")
            return None
        

  
    # predict_with_model method
    def predict_with_model(self):
        if self.model is None:
            self.result_label.setText("Please load a model first.")
            return

        if not self.image_path:
            self.result_label.setText("Please load an image first.")
            return

        img = self.preprocess_image(self.image_path)
        if img is None:
            self.result_label.setText("Error preprocessing image.")
            return

        try:
            preds = []

            # Case 1: If stacked model (e.g. LogisticRegression)
            if isinstance(self.model, LogisticRegression):
                # Load base models and get their softmax outputs
                for base_model_path in [
                    self.model_paths["Model 1"],
                    self.model_paths["Model 2"],
                    self.model_paths["Model 3"],
                    self.model_paths["Model 4"],
                    self.model_paths["Model 5"],
                    self.model_paths["Model 6"]
                ]:
                    base_model = load_model(base_model_path, compile=False)
                    prob = base_model.predict(img)[0]  # softmax output vector
                    preds.extend(prob)

                # Make prediction with stacked model
                stacked_input = np.array(preds).reshape(1, -1)
                final_pred = self.model.predict(stacked_input)[0]
                final_probs = self.model.predict_proba(stacked_input)[0]
                confidence = final_probs[final_pred]

                class_name = self.classes.get(final_pred, "Unknown")
                self.result_label.setText(f"Predicted class: {class_name}")
                self.result_label_2.setText(f"Confidence: {confidence:.2f}")

            # Case 2: If your model is a CNN (e.g. Keras model)
            else:
                predictions = self.model.predict(img)[0]
                class_idx = np.argmax(predictions)
                confidence = predictions[class_idx]

                class_name = self.classes.get(class_idx, "Unknown")
                self.result_label.setText(f"Predicted class: {class_name}")
                self.result_label_2.setText(f"Confidence: {confidence:.2f}")

        except Exception as e:
            self.result_label.setText(f"Prediction error: {e}")
            print(f"Prediction error: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageClassifier()
    window.show()
    sys.exit(app.exec_())
# Note: Ensure that the model paths and UI file path are correct and accessible.
# Note: The model paths and UI file path should be updated to the correct locations on your system.