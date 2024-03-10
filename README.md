# Using Transfer Learning and TensorFlow 2.0 for the Classification of Various Dog Breeds
In this project, we will leverage machine learning techniques to accurately identify various dog breeds. To achieve this, we will utilize data from the Kaggle dog breed identification competition (https://www.kaggle.com/c/dog-breed-identification), which comprises over 10,000 labeled images spanning 120 different dog breeds.

This particular challenge falls under the category of multi-class image classification, signifying the task of classifying multiple breeds of dogs. Distinguishing itself from binary classification, which involves categorizing one entity against another, multi-class image classification is integral to technologies such as Tesla's self-driving cars and Airbnb's automated augmentation of listing details.

Given the significance of data preparation in deep learning endeavors, our initial focus will be on transforming the provided data into a numerical format conducive to our machine learning model.
We're going to go through the following TensorFlow/Deep Learning workflow:
1) Get data ready (download from Kaggle, store, import).
2) Prepare the data (preprocessing, the 3 sets, X & y).
3) Choose and fit/train a model (TensorFlow Hub, tf.keras.applications, TensorBoard, EarlyStopping).
4) Evaluating a model (making predictions, comparing them with the ground truth labels).
5) Improve the model through experimentation (start with 1000 images, making sure it works, increase the number of images).
6) Save, sharing and reloading your model.

To preprocess our data, TensorFlow 2.x will be employed. The primary objective is to convert our data into Tensors, essentially arrays of numbers compatible with GPU processing, enabling a machine learning model to identify patterns within them.

Our machine learning model will be built upon a pretrained deep learning model sourced from TensorFlow Hub. This approach, known as transfer learning, involves taking a pretrained model and customizing it to address our specific problem. The rationale behind this choice is to capitalize on established patterns within a model that has undergone training for image classification, thereby avoiding the need to train an entirely new model from scratch, a process that can be both time-consuming and resource-intensive.
