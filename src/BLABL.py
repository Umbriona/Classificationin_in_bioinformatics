import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

import tensorflow as tf

from helper_functions import coef_det_k

os.environ["CUDA_VISIBLE_DEVICES"]="0"

embedings_file = "../results/Embeddings/embedings_all_tms_nonredun_780.pkl"
with open(embedings_file, "rb") as f:
    df = pickle.load(f)

# reformat data frame as hashmap with keys as id
data_df = {}
for id_, EMB, TM in zip(df["id"], df["Embedding"], df["TM"]):
    data_df[id_] = [EMB, TM]

## Split data in to train val test

df_all_30   = pd.read_csv("../data/clu_30_cluster.tsv", sep="\t", header=None, names=["clusters", "sequences"])
#df_all_30.columns = ["clusters", "sequence"]
df_train_30 = pd.read_csv("../data/train_clusters.tsv", sep="\t", header=None, names=["clusters"])
df_val_30   = pd.read_csv("../data/val_clusters.tsv", sep="\t", header=None, names=["clusters"])
df_test_30  = pd.read_csv("../data/test_clusters.tsv", sep="\t", header=None, names=["clusters"])

df_train_30 = df_train_30.merge(df_all_30, on = "clusters", how = "left")
df_val_30   = df_val_30.merge(df_all_30, on = "clusters", how = "left")
df_test_30  = df_test_30.merge(df_all_30, on = "clusters", how = "left")


training_data = df_train_30["sequences"].apply(lambda x: data_df[x])
val_data      = df_val_30["sequences"].apply(lambda x: data_df[x])
test_data     = df_test_30["sequences"].apply(lambda x: data_df[x])

df_train_30["Embedding"], df_train_30["TM"] = zip(*[(eli[0], eli[1]) for eli in training_data])
df_val_30["Embedding"], df_val_30["TM"]     = zip(*[(eli[0], eli[1]) for eli in val_data])
df_test_30["Embedding"], df_test_30["TM"]   = zip(*[(eli[0], eli[1]) for eli in test_data])

x_train = np.array(df_train_30["Embedding"].to_list())
y_train = np.array(df_train_30["TM"])

x_val   = np.array(df_val_30["Embedding"].to_list())
y_val   = np.array(df_val_30["TM"])

x_test = np.array(df_test_30["Embedding"].to_list())
y_test = np.array(df_test_30["TM"])

model = tf.keras.Sequential()

model.add(tf.keras.layers.Dense(1024,input_shape=(1280,), activation="linear"))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.ReLU())
model.add(tf.keras.layers.Dropout(0.4))
model.add(tf.keras.layers.Dense(1024, activation="linear"))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.ReLU())
model.add(tf.keras.layers.Dropout(0.4))
model.add(tf.keras.layers.Dense(256, activation="linear"))
model.add(tf.keras.layers.ReLU())
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(1, activation = "linear"))
model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss=tf.keras.losses.MeanSquaredError(), metrics=[ coef_det_k])
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.0001)
estop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=10,
    verbose=0,
    mode='auto',
    baseline=None,
    restore_best_weights=False,
    start_from_epoch=50
)
history = model.fit(x = x_train, y = y_train, batch_size = 128, validation_data = (x_val, y_val), epochs=100, callbacks=[reduce_lr, estop]) 
