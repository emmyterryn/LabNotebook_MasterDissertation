import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import copy

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
from scipy.stats import uniform, randint
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from IPython.display import display
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate, Dense, Flatten
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Bidirectional, LSTM, GRU
from sklearn.metrics import mean_squared_error

from scipy.stats import pearsonr
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

# Define the input shapes
matrix_shape = (1148760, 60, 6)
matrix_all_shape = (1148760, 12)

# Define input layers for each matrix
input_matrix = Input(shape=(matrix_shape[1], matrix_shape[2]), name='matrix_input')
input_matrix_all = Input(shape=(matrix_all_shape[1],), name='matrix_input_all')

# GRU layer for sequence data
gru_layer = GRU(units=64, return_sequences=True)(input_matrix)

# Flatten the output of LSTM and concatenate with non-sequence data
flattened_gru = Flatten()(gru_layer)
concatenated_outputs = Concatenate()([flattened_gru, input_matrix_all])

# Add dense layers with dropout and batch normalization
dense1 = Dense(units=1024, activation = "relu")(concatenated_outputs)
dense2 = Dense(units=1014, activation = "relu")(dense1)
dense3 = Dense(units=1024, activation = "relu")(dense2)
dense4 = Dense(units=512, activation = "relu")(dense3)
output = Dense(units=1)(dense4)

# Create the model
model_rnn_gru = Model(inputs=[input_matrix, input_matrix_all], outputs=output)

# Compile the model
model_rnn_gru.compile(optimizer='adam', loss='mae')

# Print the model summary
model_rnn_gru.summary()

# Train the model with your input and output data
history_rnn_gru = model_rnn_gru.fit([matrix_train, matrix_all_train], ccs_train, epochs=20, batch_size=24, validation_data=([matrix_val, matrix_all_val], ccs_val))

history_df_rnn_gru = pd.DataFrame(history_rnn_gru.history)
history_df_rnn_gru.to_csv('/home/emmy/Notebooks2/history_rnn_gru.csv', index=False)
print("Minimum validation loss: {}".format(history_df_rnn_gru['val_loss'].min()))

# Predict CCS values test set
ccs_test["Model_rnn_gru_predictions"] = model_rnn_gru.predict((matrix_test, matrix_all_test))

ccs_test.to_csv('/home/emmy/Notebooks2/ccs_test_model_rnn_gru.csv', index=False)

