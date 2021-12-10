

##Tensorflow implementation of checkpointing with HTCondor

import subprocess
import os
from os.path import exists
import sys
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint

##Ensure GPU usage
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


"""
Set up some checkpointing stuff
"""

##Checking if any checkpoints already exist

chkpt_exists = os.path.exists("checkpoint.txt")
if not chkpt_exists:
  chkpt = 0
else:
  with open("checkpoint.txt", 'r') as f:
    chkpt = int(f.read())
    f.close()

##Some training params are also needed for checkpointing
num_epochs = 20
remaining = num_epochs-chkpt
chkpt_frequency = 5


"""
Load the data and define the model.
"""

##Load Data
(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

##Normalize Data
def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

##Define Model
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128,activation='relu'),
  tf.keras.layers.Dense(10)
])

##Compile Model
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

"""
Begin Training/Checkpointing
"""

##If checkpoints exist, continue training
if chkpt_exists:
  if num_epochs-(chkpt + chkpt_frequency)>0: ##ie the model won't finish this round
    print("Checkpoint Found - "+str(chkpt)+"/"+str(num_epochs))
    ##Load checkpoint and keep going
    model.load_weights("checkpoint.h5")
    print("Loaded checkpoint")

    with tf.device('/GPU:0'):      
      model.fit(ds_train,
        epochs=chkpt_frequency,
        validation_data=ds_test,
      )
    model.save_weights("checkpoint.h5", overwrite=True) ##Save the checkpointed model

    with open("checkpoint.txt", 'w') as f: ##Save the current checkpoint value
      chkpt = chkpt + chkpt_frequency
      f.write(str(chkpt))
      f.close()
    sys.exit(85) ##Exit with non-zero to tell HTCondor we have more to do

  else: ##The model should finish training -- train it on the remaining epochs
    print("Checkpoint Found - "+str(chkpt)+"/"+str(num_epochs))
    ##Load checkpoint and keep going
    model.load_weights("checkpoint.h5")
    print("Loaded checkpoint")
   
    with tf.device('/GPU:0'):   
      model.fit(ds_train,
        epochs=remaining,
        validation_data=ds_test,
      )

    model_filepath = "model/"
    model.save(model_filepath) ##Save the completed model
    subprocess.run(["tar", "czvf","model.tar.gz", model_filepath]) ##Zip the model file so it can be transferred back to the submit node
    sys.exit(0) ##Exit with 0 to tell HTCondor we're done

##If no checkpoint exists, start from scratch
else:
  print("No checkpoint found \nInitializing from scratch")
  
  with tf.device('/GPU:0'):
    model.fit(
      ds_train,
      epochs=chkpt_frequency,
      validation_data=ds_test,
    )
  
  model.save_weights("checkpoint.h5") 
  with open("checkpoint.txt", 'w') as f: ##Create a file to store checkpoint value
    chkpt = chkpt + chkpt_frequency
    f.write(str(chkpt))
    f.close()
  sys.exit(85) ##Exit with non-zero to tell HTCondor we have more to do

