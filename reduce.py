#!pip install idx2numpy

import idx2numpy
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from sklearn.model_selection import train_test_split

# Load and preprocess the input data from input.dat and query.dat
# Update the file path to point to the IDX file directly
file_path = "input.dat"
query_path = "query.dat"
# Load the data
input_data = idx2numpy.convert_from_file(file_path)
query_data = idx2numpy.convert_from_file(query_path)

# Reshape the data to match the expected input shape of the autoencoder
input_data = input_data.reshape(-1, 28, 28, 1)
query_data = query_data.reshape(-1, 28, 28, 1)

# Normalize the data
input_data = input_data.astype('float32') / 255.0
query_data = query_data.astype('float32') / 255.0

# Split the dataset into training and validation sets
x_train, x_val = train_test_split(input_data, test_size=0.1, random_state=42)

# Define the autoencoder model
input_img = Input(shape=(28, 28, 1))
x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)
print(f"Layer: {encoded.name}, Output shape: {encoded.shape}")

# At this point, the representation is (4, 4, 8) i.e. 128-dimensional

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Train the autoencoder
history = autoencoder.fit(
    x_train, x_train,
    epochs=10,
    batch_size=128,
    shuffle=True,
    validation_data=(x_val, x_val)
)

# Encode the input and query data# Encode the input data
encoded_imgs = autoencoder.get_layer('max_pooling2d_2').output
encoded_imgs_model = Model(inputs=autoencoder.input, outputs=encoded_imgs)
encoded_imgs = encoded_imgs_model.predict(input_data)
encoded_querys =  encoded_imgs_model.predict(query_data)
# Print the shape of encoded_imgs
print(encoded_imgs.shape)
print(encoded_querys.shape)

encoded_imgs.astype(np.uint8).tofile('reducedinput.dat')
encoded_querys.astype(np.uint8).tofile('reducedquery.dat')
# Plot training history
# Plot training history
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()



