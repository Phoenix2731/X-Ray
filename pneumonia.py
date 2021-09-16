## Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
#import os

#os.chdir("\Users\apple\Desktop\chest_xray")

# Initialising the CNN
pneumonia_classifier = Sequential()

# Step 1 - Convolution
pneumonia_classifier.add(Conv2D(32, (3, 3), input_shape = (128, 128, 3), activation = 'relu'))

# Step 2 - Pooling
pneumonia_classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer and pooling layer
pneumonia_classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
pneumonia_classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
pneumonia_classifier.add(Flatten())

# Step 4 - Full connection
pneumonia_classifier.add(Dense(units = 128, activation = 'relu'))
pneumonia_classifier.add(Dense(units = 64, activation = 'relu'))
pneumonia_classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
pneumonia_classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images



train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255,
                                  shear_range = 0.2,
                                  zoom_range = 0.2,
                                  horizontal_flip = True)

training_set = train_datagen.flow_from_directory('train',
                                                 target_size = (128, 128),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('test',
                                            target_size = (128, 128),
                                            batch_size = 32,
                                            class_mode = 'binary')

pneumonia_classifier.fit_generator(training_set,
                         steps_per_epoch = 100,
                         epochs = 1,
                         validation_data = test_set,
                         validation_steps = 50) 


# Part 3 - Making new predictions
import numpy as np
from keras.preprocessing import image
test_image = image.load_img(r"/Users/apple/Desktop/person1_bacteria_2.jpeg", target_size = (128, 128))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = pneumonia_classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    print('\nEFFECTED')
else:
    print('\nUNEFFECTED')

