# MNIST Digits Classification

This repository provides a comprehensive guide for implementing a multiclass image classification model using TensorFlow’s MNIST Digits Dataset. The primary goal is to develop an efficient machine learning model capable of classifying handwritten digits from 0 to 9.

## Objectives

    Develop a model to recognize and classify handwritten digits.
    Automate and enhance processes such as digital document analysis and data entry.

## Dataset Overview

- Source: MNIST dataset
- Classes: Ten distinct digit classes (0-9)
- Dataset Size:
    - Training: 60,000 images
    - Testing: 10,000 images
- Image Characteristics: Grayscale images of size 28x28 pixels.

## Model Building Steps

1. **Install Required Packages**:

```
   pip install tensorflow tensorflow-datasets opencv-python
```
2. **Import Libraries**:

```
   import numpy as np
   import matplotlib.pyplot as plt
   from tensorflow.keras.layers import Flatten, Dense, Dropout
   from keras.callbacks import EarlyStopping
   from tensorflow.keras.optimizers import RMSprop
   from sklearn.model_selection import train_test_split
   import cv2
```

3. **Data Loading and Inspection**:

```
   (X_train, y_train), (X_test, y_test) = mnist.load_data()
   print('Train: X=%s, y=%s' % (X_train.shape, y_train.shape))
   print('Test: X=%s, y=%s' % (X_test.shape, y_test.shape))
```

4. **Model Architecture**:
    - Neural network designed to achieve high accuracy in classifying handwritten digits.

5. **Evaluation**:
    -Utilized holdout validation to ensure model performance on unseen data.
    -Achieved a validation accuracy of approximately 97.57%.

## Success Metrics

- High accuracy is critical as it ensures reliability in applications like document analysis and data entry.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

By following this guide, users can better understand and implement multiclass image classification on the MNIST dataset using TensorFlow. Enjoy experimenting and improving the model!
