import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" # Uncomment this if you want to suppress TensorFlow warnings

import numpy as np
import cv2
from glob import glob
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import pandas as pd # Import pandas for CSV handling
from patchify import patchify # Make sure this is installed: pip install patchify
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
from vit import ViT # Make sure this import is correct for your setup

""" Hyperparameters """
hp = {}
hp["image_size"] = 224 # Make sure this matches your image preprocessing
hp["num_channels"] = 3
hp["patch_size"] = 25
hp["num_patches"] = (hp["image_size"] // hp["patch_size"])**2 # Correct calculation for num_patches
hp["flat_patches_shape"] = (hp["num_patches"], hp["patch_size"] * hp["patch_size"] * hp["num_channels"])

hp["batch_size"] = 32
hp["lr"] = 1e-4
hp["num_epochs"] = 3 # Set to a reasonable number, EarlyStopping will manage it
# hp["num_classes"] and hp["class_names"] will be set dynamically from CSV

hp["num_layers"] = 12
hp["hidden_dim"] = 768
hp["mlp_dim"] = 3072
hp["num_heads"] = 12
hp["dropout_rate"] = 0.1

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# --- MODIFIED: load_data to use CSV for labels and handle splitting ---
def load_data_from_csv(images_dir, labels_csv_path, test_split_ratio=0.1, val_split_ratio=0.1):
    """
    Loads image paths and labels from a CSV file, then performs a stratified 80/10/10 split.

    Args:
        images_dir (str): Path to the directory containing all image files.
        labels_csv_path (str): Path to the CSV file containing image IDs and labels.
        test_split_ratio (float): Fraction of data for the test set (e.g., 0.1 for 10%).
        val_split_ratio (float): Fraction of data for the validation set (e.g., 0.1 for 10%).

    Returns:
        tuple: (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels),
               unique_class_names, label_to_int_map
    """
    print(f"Loading labels from CSV: {labels_csv_path}")
    df_labels = pd.read_csv(labels_csv_path)

    all_image_paths = []
    all_labels = []

    # Adjust these column names if your CSV is different for image ID and label
    ID_COLUMN = 'id_code'      # e.g., 'image_id', 'filename'
    LABEL_COLUMN = 'diagnosis' # e.g., 'label', 'class'

    # Determine image file extension (assuming consistency)
    # You might need to adjust this if your dataset has mixed extensions
    sample_img_id = df_labels[ID_COLUMN].iloc[0]
    # Check common extensions
    if os.path.exists(os.path.join(images_dir, f"{sample_img_id}.png")):
        img_extension = ".png"
    elif os.path.exists(os.path.join(images_dir, f"{sample_img_id}.jpg")):
        img_extension = ".jpg"
    elif os.path.exists(os.path.join(images_dir, f"{sample_img_id}.jpeg")):
        img_extension = ".jpeg"
    else:
        # Fallback or raise error if extension not found
        print("Warning: Image extension not automatically detected. Assuming .png.")
        img_extension = ".png" # Default to .png if not found

    for _, row in df_labels.iterrows():
        image_id = row[ID_COLUMN]
        img_filename = f"{image_id}{img_extension}"
        img_path = os.path.join(images_dir, img_filename)

        if os.path.exists(img_path):
            all_image_paths.append(img_path)
            # Ensure label is a string for consistent mapping
            all_labels.append(str(row[LABEL_COLUMN]))
        # else: # Uncomment for detailed warnings if images are missing
            # print(f"Warning: Image file not found for ID '{image_id}' at '{img_path}'. Skipping.")

    if not all_image_paths:
        raise ValueError(f"No images found in '{images_dir}' matching IDs and extension from '{labels_csv_path}'. "
                         "Please check your paths, CSV columns (ID_COLUMN), and image extensions.")

    # Get unique labels and create a mapping to integers
    unique_labels = sorted(list(set(all_labels)))
    label_to_int = {label: i for i, label in enumerate(unique_labels)}

    # Map string labels to integers for stratification
    all_int_labels = [label_to_int[label] for label in all_labels]

    # --- Perform 80/10/10 split ---
    # First split: Separate 'test_split_ratio' for the final test set
    # The remaining (1 - test_split_ratio) will be used for training and validation
    train_val_x, test_x, train_val_y, test_y = train_test_split(
        all_image_paths, all_int_labels,
        test_size=test_split_ratio,
        random_state=42,
        stratify=all_int_labels
    )

    # Second split: From the (train+val) set, separate 'val_split_ratio' for validation
    # The 'test_size' for this split is relative to 'train_val_x'
    # To get `val_split_ratio` of the *original* total as validation, from the remaining set:
    # (val_split_ratio / (1 - test_split_ratio))
    val_relative_size = val_split_ratio / (1.0 - test_split_ratio)
    train_x, valid_x, train_y, valid_y = train_test_split(
        train_val_x, train_val_y,
        test_size=val_relative_size,
        random_state=42,
        stratify=train_val_y
    )
    print(f"\n--- Dataset Split Information ---")
    print(f"Total collected images: {len(all_image_paths)}")
    print(f"Train samples: {len(train_x)} ({len(train_x)/len(all_image_paths)*100:.2f}%)")
    print(f"Validation samples: {len(valid_x)} ({len(valid_x)/len(all_image_paths)*100:.2f}%)")
    print(f"Test samples: {len(test_x)} ({len(test_x)/len(all_image_paths)*100:.2f}%)")
    print(f"Number of unique classes: {len(unique_labels)}")
    print(f"Class names: {unique_labels}")
    print(f"Label to int mapping: {label_to_int}")
    print(f"---------------------------------")

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y), unique_labels, label_to_int

# --- MODIFIED: process_image_label to accept label_int directly ---
def process_image_label(path, label_int):
    """
    Reads image, applies patching, and returns patches and integer label.
    Now receives the label directly as an integer.
    """
    # Decode tensor string to python string
    path = path.decode()
    image = cv2.imread(path, cv2.IMREAD_COLOR)

    if image is None: # Handle cases where image might not load (e.g., corrupt file, wrong path)
        tf.print(f"Error: Could not read image {path}. Returning zero array.", output_stream=tf.io.gfile.GFile('/dev/stderr', 'w'))
        # Return an empty/zero tensor that matches expected shape to prevent crash
        # This might cause issues if many images fail, better to filter them out earlier
        return np.zeros(hp["flat_patches_shape"], dtype=np.float32), np.array(0, dtype=np.int32)

    image = cv2.resize(image, (hp["image_size"], hp["image_size"]))
    image = image / 255.0 # Normalize image to [0, 1]

    """ Preprocessing to patches """
    patch_shape = (hp["patch_size"], hp["patch_size"], hp["num_channels"])
    # patchify expects (H, W, C), so image is fine directly
    patches = patchify(image, patch_shape, hp["patch_size"])

    # Reshape patches to a flat array per patch (num_patches, patch_flattened_dim)
    patches = np.reshape(patches, hp["flat_patches_shape"])
    patches = patches.astype(np.float32)

    return patches, label_int # Use the integer label passed directly

# --- MODIFIED: parse to accept two arguments (path, label_int) ---
def parse(path, label_int):
    # tf.numpy_function needs to know the input types and output types
    patches, labels = tf.numpy_function(
        process_image_label,
        [path, label_int], # Pass both path and label_int to the numpy function
        [tf.float32, tf.int32]
    )
    # Convert integer label to one-hot encoding
    labels = tf.one_hot(labels, hp["num_classes"]) # hp["num_classes"] will be set after loading data

    # Set shapes for TensorFlow graph compilation
    patches.set_shape(hp["flat_patches_shape"])
    labels.set_shape(hp["num_classes"])

    return patches, labels

# --- MODIFIED: tf_dataset to accept (paths, labels) ---
def tf_dataset(paths, labels, batch=32):
    # Create dataset from both image paths and their integer labels
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    # Map the parse function to each (path, label) pair
    ds = ds.map(parse, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch).prefetch(tf.data.AUTOTUNE) # Use tf.data.AUTOTUNE for prefetch buffer size
    return ds


if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files """
    create_dir("files")

    """ Paths (for your dataset) """
    # IMPORTANT: Adjust these paths to your actual image folder and CSV file
    # For APTOS, this is typical:
    dataset_images_dir = "C:\\Users\\Malek\\PycharmProjects\\Latif\\APTOS_mini\\train_images" # Folder containing image files (e.g., 000a6adce.png)
    dataset_labels_csv = "C:\\Users\\Malek\\PycharmProjects\\Latif\\APTOS_mini\\train_images.csv"   # CSV file with image_id and diagnosis

    model_path = os.path.join("files", "model.keras")
    csv_path = os.path.join("files", "log.csv")

    """ Dataset """
    # Call load_data_from_csv with image directory and CSV path
    # It returns tuples of (paths, labels) for each set, plus class info
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y), unique_class_names, label_int_map = \
        load_data_from_csv(dataset_images_dir, dataset_labels_csv)

    # Dynamically set num_classes and class_names based on loaded data
    hp["num_classes"] = len(unique_class_names)
    hp["class_names"] = unique_class_names

    # Create TensorFlow datasets
    train_ds = tf_dataset(train_x, train_y, batch=hp["batch_size"])
    valid_ds = tf_dataset(valid_x, valid_y, batch=hp["batch_size"])
    test_ds = tf_dataset(test_x, test_y, batch=hp["batch_size"]) # Create test_ds here

    """ Model """
    model = ViT(hp) # Pass the updated hp dictionary to ViT
    model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(hp["lr"], clipvalue=1.0),
        metrics=["acc"]
    )
    model.summary() # Good to see the model architecture and parameters

    callbacks = [
        ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=1e-10, verbose=1),
        CSVLogger(csv_path),
        EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True), # restore_best_weights=True is generally recommended
    ]

    print("\nStarting model training...")
    model.fit(
        train_ds,
        epochs=hp["num_epochs"],
        validation_data=valid_ds,
        callbacks=callbacks
    )

    # Evaluate the model on the test set after training
    print("\nEvaluating model on test set...")
    # Load the best saved model. Make sure ViT class is available for loading.
    try:
        best_model = tf.keras.models.load_model(model_path, custom_objects={'ViT': ViT})
    except Exception as e:
        print(f"Could not load best model. Error: {e}. Using the last trained model.")
        best_model = model

    test_loss, test_acc = best_model.evaluate(test_ds, verbose=1)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc*100:.2f}%")