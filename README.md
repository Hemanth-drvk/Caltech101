Caltech-101 Image Classification This code implements a Convolutional
Neural Network (CNN) for image classification on the Caltech-101
dataset. The code uses Keras with TensorFlow backend.

Prerequisites Python 3.6 or higher Keras 2.4.3 or higher TensorFlow
2.4.1 or higher scipy matplotlib pandas scikit-learn Installation Clone
the repository and install the required packages.

bash Copy code git clone https://github.com/<username>/<repository>.git
cd <repository> pip install -r requirements.txt Usage Download the
Caltech-101 dataset from
http://www.vision.caltech.edu/Image_Datasets/Caltech101/. Extract the
dataset to a folder named 101_ObjectCategories.

Open the Jupyter notebook caltech-101.ipynb.

Run the notebook to train and evaluate the CNN model.

Note: The path to the 101_ObjectCategories folder can be changed in the
load_dataset function.

Acknowledgments This project was inspired by the book “Deep Learning
with Python” by Francois Chollet. The Caltech-101 dataset was collected
by Fei-Fei Li, Marco Andreetto, and Marc ’Aurelio Ranzato.
