
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import cv2
from patchify import patchify
import tensorflow as tf
from train import load_data_from_csv, tf_dataset
from vit import ViT

""" Hyperparameters """
hp = {}
hp["image_size"] = 200
hp["num_channels"] = 3
hp["patch_size"] = 25
hp["num_patches"] = (hp["image_size"]**2) // (hp["patch_size"]**2)
hp["flat_patches_shape"] = (hp["num_patches"], hp["patch_size"]*hp["patch_size"]*hp["num_channels"])

hp["batch_size"] = 16
hp["lr"] = 1e-4
hp["num_epochs"] = 500
hp["num_classes"] = 5
hp["class_names"] = ["0", "1", "2", "3", "4"]

hp["num_layers"] = 12
hp["hidden_dim"] = 768
hp["mlp_dim"] = 3072
hp["num_heads"] = 12
hp["dropout_rate"] = 0.1


if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Paths """
    dataset_path = "C:\\Users\\Malek\\PycharmProjects\\Latif\\APTOS_mini\\train_images"
    dataset_path_csv = "C:\\Users\\Malek\\PycharmProjects\\Latif\\APTOS_mini\\train_images.csv"

    model_path = os.path.join("files", "model.keras")

    """ Dataset """
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y), unique_class_names, label_int_map = \
        load_data_from_csv(dataset_path, dataset_path_csv)
    print(f"Train: {len(train_x)} - Valid: {len(valid_x)} - Test: {len(test_x)}")

    test_ds = tf_dataset(test_x,test_y, batch=hp["batch_size"])

    """ Model """
    model = ViT(hp)
    model.load_weights(model_path)
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        optimizer=tf.keras.optimizers.Adam(hp["lr"]),
        metrics=["acc"]
    )

    model.evaluate(test_ds)
