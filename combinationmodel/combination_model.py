import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import copy

from psm_utils.io.peptide_record import peprec_to_proforma
from psm_utils.psm import PSM
from psm_utils.psm_list import PSMList
from psm_utils.io import write_file
from deeplc import FeatExtractor



from sklearn.metrics import mean_absolute_error
from scikeras.wrappers import KerasRegressor
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras import callbacks
from sklearn.model_selection import RandomizedSearchCV, KFold
from scipy.stats import uniform, randint, pearsonr
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from IPython.display import display
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate, Dense, Flatten, Attention
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Bidirectional, LSTM
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from sklearn.metrics import mean_squared_error

#encoding by atomic composition 

infile = pd.read_csv("/home/emmy/Notebooks2/MQ_alignment_output/evidence_aligned_6.csv") #hier eigen aligned csv file aanroepen
infile.head()
psm_list = [] #psm_list is type object 
for idx,row in infile.iterrows():
    seq = row["Sequence"]
    charge = row["Charge"]  # Get the charge from the row

    peptidoform = f"{seq}/{charge}"

    psm_list.append(PSM(peptidoform=peptidoform,spectrum_id=idx))
    
psm_list = PSMList(psm_list=psm_list)

feat_extractor = FeatExtractor()
matrices = feat_extractor.encode_atoms(psm_list, list(range(len(psm_list))), predict_ccs=True)

data = pd.read_csv("/home/emmy/Notebooks2/MQ_alignment_output/evidence_aligned_6.csv") #reading in the data
ccs_df = data[['CCS']]

matrix = np.stack(list(matrices["matrix"].values()))
matrix_all = np.stack(list(matrices["matrix_all"].values()))

# Split the data into training and testing sets
matrix_train, matrix_test, matrix_all_train, matrix_all_test, ccs_train, ccs_test = train_test_split(matrix, matrix_all, ccs_df, test_size=0.2, random_state=42)

# Further split the training data into training and validation sets if needed
matrix_train, matrix_val, matrix_all_train, matrix_all_val, ccs_train, ccs_val = train_test_split(matrix_train, matrix_all_train, ccs_train, test_size=0.1, random_state=42)

print("data ready")
#add early stopping
early_stopping = callbacks.EarlyStopping(
    min_delta=0.001, # minimium amount of change to count as an improvement
    patience=20, # how many epochs to wait before stopping
    restore_best_weights=True,
)

# Define the input shapes
matrix_shape = (1148760, 60, 6)
matrix_all_shape = (1148760, 12)

# Define input layers for each matrix
input_matrix = Input(shape=(matrix_shape[1], matrix_shape[2]), name='matrix_input')
input_matrix_all = Input(shape=(matrix_all_shape[1],), name='matrix_input_all')

# Bidirectional LSTM layer for sequence data
lstm_layer = Bidirectional(LSTM(units=200, return_sequences=True))(input_matrix)

# Apply attention mechanism to the output of LSTM
attention = Attention()([lstm_layer, lstm_layer])

# Flatten the output of attention mechanism and concatenate with non-sequence data
flattened_attention = Flatten()(attention)
concatenated_outputs = Concatenate()([flattened_attention, input_matrix_all])

# Add dense layers with dropout and batch normalization
dense1 = Dense(units=1024, activation="relu")(concatenated_outputs)
dense2 = Dense(units=1014, activation = "relu")(dense1)
dense3 = Dense(units=1024, activation = "relu")(dense2)
dense4 = Dense(units=512, activation = "relu")(dense3)
output = Dense(units=1)(dense4)  # Output layer

# Create the model
model_rnn9 = Model(inputs=[input_matrix, input_matrix_all], outputs=output)

# Compile the model
model_rnn9.compile(optimizer=Adam(learning_rate = 0.0003), loss='mae')

#add early stopping
early_stopping = callbacks.EarlyStopping(
    min_delta=0.001, # minimium amount of change to count as an improvement
    patience=20, # how many epochs to wait before stopping
    restore_best_weights=True,
)

# Define the input shapes
matrix_shape = (1148760, 60, 6)
matrix_all_shape = (1148760, 12)

# Define input layers for each matrix
input_matrix_cnn = Input(shape=(matrix_shape[1], matrix_shape[2]), name='matrix_input_cnn')
input_matrix_all_cnn = Input(shape=(matrix_all_shape[1],), name='matrix_input_all_cnn')

# First Conv1D layer
conv1d_layer1 = Conv1D(filters=64, kernel_size=5, activation='relu')(input_matrix_cnn)
maxpooling_layer1 = MaxPooling1D(pool_size=2)(conv1d_layer1)

# Second Conv1D layer
conv1d_layer2 = Conv1D(filters=64, kernel_size=3, activation='relu')(maxpooling_layer1)
maxpooling_layer2 = MaxPooling1D(pool_size=2)(conv1d_layer2)

# Third Conv1D layer
conv1d_layer3 = Conv1D(filters=100, kernel_size=3, activation='relu')(maxpooling_layer2)
maxpooling_layer3 = MaxPooling1D(pool_size=2)(conv1d_layer3)

# Global max pooling layer
globalmaxpooling_layer = GlobalMaxPooling1D()(maxpooling_layer3)

# Concatenate with non-sequence data
concatenated_outputs = Concatenate()([globalmaxpooling_layer, input_matrix_all_cnn])

# Add dense layers 
dense1 = Dense(units=1024, activation = "relu")(concatenated_outputs)
dense2 = Dense(units=1014, activation = "relu")(dense1)
dense3 = Dense(units=1024, activation = "relu")(dense2)
dense4 = Dense(units=512, activation = "relu")(dense3)
output = Dense(units=1)(dense4)

# Create the model
model_cnn7 = Model(inputs=[input_matrix_cnn, input_matrix_all_cnn], outputs=output)

# Compile the model
model_cnn7.compile(optimizer=Adam(learning_rate = 0.0003), loss='mae')

# Combine the output layers of the RNN and CNN models
from keras.layers import Concatenate, Dense

combined_output = Concatenate()([model_rnn9.output, model_cnn7.output])

# Add additional dense layers for prediction
dense1 = Dense(units=512, activation="relu")(combined_output)
dense2 = Dense(units=256, activation="relu")(dense1)
output = Dense(units=1)(dense2)  # Output layer

# Create the parallel model
from keras.models import Model
from keras.optimizers import Adam

parallel_model = Model(
    inputs=[model_rnn9.input, model_cnn7.input],
    outputs=output
)

# Compile the parallel model
parallel_model.compile(optimizer=Adam(learning_rate=0.0003), loss='mae')

# Print the model summary
parallel_model.summary()

history_combined = parallel_model.fit(
    [matrix_train, matrix_all_train, matrix_train, matrix_all_train], 
    ccs_train, 
    epochs=100, 
    batch_size=256, 
    validation_data=([matrix_val, matrix_all_val, matrix_val, matrix_all_val], ccs_val), 
    callbacks=[early_stopping]
)

history_df_combined = pd.DataFrame(history_combined.history)
history_df_combined.to_csv('/home/emmy/Notebooks2/history_combined.csv', index=False)
print("Minimum validation loss: {}".format(history_df_combined['val_loss'].min()))

# Predict CCS values test set
ccs_test["Model_combined_predictions"] = parallel_model.predict([matrix_test, matrix_all_test, matrix_test, matrix_all_test])

ccs_test.to_csv('/home/emmy/Notebooks2/ccs_test_model_combined.csv', index=False)