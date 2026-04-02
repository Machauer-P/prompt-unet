import os
import tensorflow as tf
import pickle

class DSHandler:
    """Class to save and load TF datasets as TFRecords, with path passed to each method."""

    # ----------------- Saving Methods ----------------- #
    def save_initial_ds(self, ds: dict, filename: str):
        """Save a dictionary as a pickle file."""
        with open(filename + ".pkl", "wb") as f:
            pickle.dump(ds, f)         
                    
    def save_tf_dataset_volume(self, tf_ds, filename, path="."):
        """
        Save a dataset of (x, y) tuples to a TFRecord file.
        Optimized: batches for speed, writes each example individually.
        """
        os.makedirs(path, exist_ok=True)
        filepath = os.path.join(path, filename)

        def serialize_example(x, y):
            feature = {
                'x': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(x).numpy()])),
                'y': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(y).numpy()])),
            }
            example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
            return example_proto.SerializeToString()

        with tf.io.TFRecordWriter(filepath) as writer:
            for x, y in tf_ds:  # no batching
                writer.write(serialize_example(x, y))


    def save_tf_dataset_2D(self, tf_ds, filename, path=".", batch_size=32):
        """
        Save a dataset of (x, y, p) tuples to a TFRecord file.
        Optimized: batches for speed, writes each example individually.
        """
        os.makedirs(path, exist_ok=True)
        filepath = os.path.join(path, filename)

        def serialize_example(x, y, p):
            feature = {
                'x': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(x).numpy()])),
                'y': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(y).numpy()])),
                'p': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(p).numpy()])),
            }
            example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
            return example_proto.SerializeToString()

        tf_ds_batched = tf_ds.batch(batch_size)

        with tf.io.TFRecordWriter(filepath) as writer:
            for batch in tf_ds_batched:
                x_batch, y_batch, p_batch = batch
                for x, y, p in zip(x_batch, y_batch, p_batch):
                    writer.write(serialize_example(x, y, p))
                    
                    
    # ----------------- Loading Methods ----------------- #
    def load_tf_dataset_volume(self, filename, path="."):
        """Load a dataset of (x, y) from a TFRecord file in the specified path."""
        filepath = os.path.join(path, filename)
        feature_description = {
            'x': tf.io.FixedLenFeature([], tf.string),
            'y': tf.io.FixedLenFeature([], tf.string),
        }

        def _parse_function(example_proto):
            parsed = tf.io.parse_single_example(example_proto, feature_description)
            x = tf.io.parse_tensor(parsed['x'], out_type=tf.float32)
            y = tf.io.parse_tensor(parsed['y'], out_type=tf.float32)
            return x, y

        return tf.data.TFRecordDataset(filepath).map(_parse_function)


    def load_tf_dataset_2D(self, filename, path="."):
        """Load a dataset of (x, y, p) from a TFRecord file in the specified path."""
        filepath = os.path.join(path, filename)
        feature_description = {
            'x': tf.io.FixedLenFeature([], tf.string),
            'y': tf.io.FixedLenFeature([], tf.string),
            'p': tf.io.FixedLenFeature([], tf.string),
        }

        def _parse_function(example_proto):
            parsed = tf.io.parse_single_example(example_proto, feature_description)
            x = tf.io.parse_tensor(parsed['x'], out_type=tf.float32)
            y = tf.io.parse_tensor(parsed['y'], out_type=tf.float32)
            p = tf.io.parse_tensor(parsed['p'], out_type=tf.float32)
            return x, y, p

        return tf.data.TFRecordDataset(filepath).map(_parse_function)
