import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
from scipy.stats import uniform, randint, pearsonr
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from IPython.display import display
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate, Dense, Flatten
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Bidirectional, LSTM, GRU, Attention
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D
from sklearn.metrics import mean_squared_error
from tensorflow.keras.optimizers import Adam

infile = pd.read_csv("/home/emmy/Notebooks2/output/MQpeprec_1.csv") #hier peprec aanroepen

infile.head()
psm_list = [] #psm_list is type object 
for idx,row in infile.iterrows():
    seq = row["sequence"]
    charge = row["charge"]  # Get the charge from the row
    mod = row["modifications"]
    
    peptidoform = f"{seq}/{charge}"
    psm_list.append(PSM(peptidoform=peprec_to_proforma(peptidoform,mod),spectrum_id=idx))
    
psm_list = PSMList(psm_list=psm_list)

feat_extractor = FeatExtractor()
matrices = feat_extractor.encode_atoms(psm_list, list(range(len(psm_list))), predict_ccs=True)

data = pd.read_csv("/home/emmy/Notebooks2/output/MQpeprec_1.csv") #reading in the data
ccs_df = data[['tr']]
ccs_df = ccs_df.rename(columns={'tr': 'CCS'})

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
dense1 = Dense(units=1024, activation = "relu")(concatenated_outputs)
dense2 = Dense(units=1014, activation = "relu")(dense1)
dense3 = Dense(units=1024, activation = "relu")(dense2)
dense4 = Dense(units=512, activation = "relu")(dense3)
output = Dense(units=1)(dense4)

# Create the model
model_rnn_mod9 = Model(inputs=[input_matrix, input_matrix_all], outputs=output)

# Compile the model
model_rnn_mod9.compile(optimizer=Adam(learning_rate = 0.0003), loss='mae')

# Print the model summary
model_rnn_mod9.summary()

# Train the model with your input and output data
history_rnn_mod9 = model_rnn_mod9.fit([matrix_train, matrix_all_train], ccs_train, epochs=100, batch_size=256, validation_data=([matrix_val, matrix_all_val], ccs_val), callbacks=[early_stopping])

history_df_rnn_mod9= pd.DataFrame(history_rnn_mod9.history)
history_df_rnn_mod9.to_csv('/home/emmy/Notebooks2/history_rnn_mod9.csv', index=False)
print("Minimum validation loss: {}".format(history_df_rnn_mod9['val_loss'].min()))

# Predict CCS values test set
ccs_test["Model_rnn_mod9_predictions"] = model_rnn_mod9.predict((matrix_test, matrix_all_test))

ccs_test.to_csv('/home/emmy/Notebooks2/ccs_test_model_rnn_mod9.csv', index=False)