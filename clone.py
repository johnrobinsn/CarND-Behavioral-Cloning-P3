import csv
import cv2
import numpy as np
from math import ceil

samples = []
    
def load_samples(base_dir):
    global samples
    
    base_img_dir = base_dir + '/IMG/'
    driving_log = base_dir+'/driving_log.csv'

    # Load and expand dataset
    with open(driving_log) as csvfile:
        reader = csv.reader(csvfile)
        next(reader) #skip first line
        for line in reader:
            img_center = base_img_dir+line[0].split("/")[-1]
            img_left = base_img_dir+line[1].split("/")[-1]
            img_right = base_img_dir+line[2].split("/")[-1]

            steering_center = float(line[3])

            # create adjusted steering measurements for the side camera images
            correction = 0.20 # this is a parameter to tune
            steering_left = steering_center + correction
            steering_right = steering_center - correction

            samples.append((img_center,steering_center,False)) # image_path, measurement, flip

            # if the steering angle exceeds a certain threshold use the views from the left and right
            # camera to train the model to accentuate the training signal.
            if abs(steering_center) > 0.33:
                samples.append((img_left,steering_left,False))
                samples.append((img_right,steering_right,False))
                
# Load recorded data from a training run
# Can load one training run after another
load_samples('/opt/data')
#load_samples('/opt/track2')
        
# Generate more data by flipping the images horizontally and negating the steering measurements.
augmented_samples = []
for sample in samples:
    augmented_samples.append(sample)
    augmented_samples.append((sample[0],-sample[1],True))     
        
from sklearn.model_selection import train_test_split

# Split the data set into training and validation sets
train_samples, validation_samples = train_test_split(augmented_samples, test_size=0.2)

from sklearn.utils import shuffle

# Generator used to feed in data during training and validation
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = batch_sample[0]
                image = cv2.imread(name)
                angle = float(batch_sample[1])
                images.append(image if not batch_sample[2] else cv2.flip(image,1))
                angles.append(angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

# Set our batch size
batch_size=100

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

# Create the model used for training
model = Sequential()
model.add(Lambda(lambda x: x/255.0-0.5, input_shape=(160,320,3))) # normalize the image data
model.add(Cropping2D(cropping=((70,25), (0,0)))) # crop the top and bottom of the input images
model.add(Convolution2D(6,5,5,activation='relu')) 
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5,activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5,activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.75))
model.add(Flatten())
model.add(Dense(128))
model.add(Dense(84))
model.add(Dense(1))  # only need a single steering output

model.compile(loss='mse', optimizer='adam')

model.fit_generator(train_generator, 
            steps_per_epoch=ceil(len(train_samples)/batch_size), 
            validation_data=validation_generator, 
            validation_steps=ceil(len(validation_samples)/batch_size), 
            epochs=5, verbose=1)


model.save('model.h5')
    
    