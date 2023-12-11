import pandas as pd
import numpy as np
import tensorflow as tf
from tensorboard.plugins import projector
import os

# Load the embeddings
embeddings_df = pd.read_csv("data/wordembed/glove_embedding.csv", index_col=0)
embeddings = embeddings_df.values
words = embeddings_df.index.tolist()

# Path to save the embeddings and metadata for TensorBoard
log_dir = "logs/embedding"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

with open(os.path.join(log_dir, 'metadata.tsv'), 'w') as f:
    for word in words:
        f.write(f"{word}\n")

# Create a TensorFlow variable with your embeddings
embedding_var = tf.Variable(embeddings, name='glove_embedding')

writer = tf.summary.create_file_writer(log_dir)

config = projector.ProjectorConfig()
embedding_config = config.embeddings.add()
embedding_config.tensor_name = embedding_var.name
embedding_config.metadata_path = 'metadata.tsv'  # Path to your metadata file

projector_config_path = os.path.join(log_dir, 'projector_config.pbtxt')
with open(projector_config_path, 'w') as f:
    f.write(str(config))

ckpt = tf.train.Checkpoint(embedding=embedding_var)
ckpt_manager = tf.train.CheckpointManager(ckpt, log_dir, max_to_keep=1)
ckpt_manager.save()

# have to load metadata.csv manually on the platform