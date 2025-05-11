import os
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from collections import Counter

from tensorflow.keras.applications import VGG16, ResNet50, EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import Adam


from sklearn.metrics import classification_report




# --- Configuration ---
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 5
RANDOM_SEED = 42


#---Chemins d'acces---
CSV_PATH = 'C:/Users/Malek/Downloads/APTOS_mini/train_images.csv'
IMAGE_DIR = 'C:/Users/Malek/Downloads/APTOS_mini/train_images'


# --- Fonctions utilitaires ---
def load_and_preprocess_image(image_path, target_size=IMAGE_SIZE):
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize(target_size)
        img_array = np.array(img) / 255.0
        return img_array
    except Exception as e:
        print(f"[Erreur] {image_path} → {e}")
        return None

def load_data_from_csv(csv_path, image_dir, target_size=IMAGE_SIZE):
    df = pd.read_csv(csv_path)
    images, labels = [], []

    for _, row in df.iterrows():
        image_id = str(row['id_code'])
        label = row['diagnosis']
        possible_exts = ['.png', '.jpg', '.jpeg']
        found = False

        for ext in possible_exts:
            image_path = os.path.join(image_dir, image_id + ext)
            if os.path.exists(image_path):
                img_array = load_and_preprocess_image(image_path, target_size)
                if img_array is not None:
                    images.append(img_array)
                    labels.append(label)
                found = True
                break

        if not found:
            print(f"[Avertissement] Image introuvable : {image_id}")

    return np.array(images), np.array(labels)


# --- Modèles ---
def create_vgg16_model(input_shape, num_classes):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = False
    x = Flatten()(base_model.output)
    x = Dense(256, activation='relu')(x)
    output = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=base_model.input, outputs=output)

def create_resnet50_model(input_shape, num_classes):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = False
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(256, activation='relu')(x)
    output = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=base_model.input, outputs=output)

def create_efficientnetb0_model(input_shape, num_classes):
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = False
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(256, activation='relu')(x)
    output = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=base_model.input, outputs=output)

def create_vit_capsnet_placeholder(input_shape, num_classes):
    input_tensor = Input(shape=input_shape)
    x = Flatten()(input_tensor)
    x = Dense(128, activation='relu')(x)
    output = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=input_tensor, outputs=output, name='vit_capsnet')
    print("⚠️ Placeholder ViT/CapsNet utilisé.")
    return model




## Graphiques

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score


def save_training_curves(history, model_name):
    acc = history.history.get('accuracy', [])
    val_acc = history.history.get('val_accuracy', [])
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])

    # Courbe de précision
    plt.figure()
    plt.plot(acc, label='Train Accuracy')
    plt.plot(val_acc, label='Val Accuracy')
    plt.title(f'Accuracy - {model_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'Graphiques/{model_name}_accuracy.png')
    plt.close()

    # Courbe de perte
    plt.figure()
    plt.plot(loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.title(f'Loss - {model_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'Graphiques/{model_name}_loss.png')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, labels, model_name="ensemble"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f'Matrice de confusion - {model_name}')
    plt.xlabel('Prédictions')
    plt.ylabel('Réel')
    plt.tight_layout()
    os.makedirs("Graphiques", exist_ok=True)
    plt.savefig(f'Graphiques/confusion_matrix_{model_name}.png')
    plt.show()



# --- Chargement des données ---
print("🔍 Chargement des données depuis CSV...")
images, labels = load_data_from_csv(CSV_PATH, IMAGE_DIR)
print(f"✅ {len(images)} images chargées.")
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- Prétraitement des labels ---
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
NUM_CLASSES = len(np.unique(encoded_labels))  # Auto-calculé si oublié
categorical_labels = tf.keras.utils.to_categorical(encoded_labels, num_classes=NUM_CLASSES)
class_names = [str(c) for c in label_encoder.classes_]

# --- Split des données ---
train_images, test_images, train_labels, test_labels = train_test_split(
    images, categorical_labels, test_size=0.2, random_state=RANDOM_SEED, stratify=categorical_labels
)

# --- Initialisation des modèles ---
input_shape = IMAGE_SIZE + (3,)
models = [
    create_vgg16_model(input_shape, NUM_CLASSES),
    create_resnet50_model(input_shape, NUM_CLASSES),
    create_efficientnetb0_model(input_shape, NUM_CLASSES),
    create_vit_capsnet_placeholder(input_shape, NUM_CLASSES)
]
model_names = ["VGG16", "ResNet50", "EfficientNetB0", "ViT_CapsNet"]
predictions = []

# --- Décodage labels de test (utile pour class_weight & évaluation) ---
true_labels = np.argmax(test_labels, axis=1)
decoded_true_labels = label_encoder.inverse_transform(true_labels)

# --- Calcul des poids de classes ---
all_train_labels = np.argmax(train_labels, axis=1)
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(all_train_labels), y=all_train_labels)
class_weight_dict = dict(enumerate(class_weights))

# --- Entraînement + prédiction ---
for i, model in enumerate(models):
    print(f"\n📚 Entraînement du modèle : {model_names[i]}")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),  # learning rate plus raisonnable
                  loss='categorical_crossentropy',
                  metrics=['accuracy', 'AUC', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])

    if model.name == 'vit_capsnet':
        history = model.fit(np.random.rand(*train_images.shape), train_labels,
                            epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1,
                            verbose=0, shuffle=True, class_weight=class_weight_dict)
    else:
        history = model.fit(train_images, train_labels,
                            epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1,
                            verbose=1, shuffle=True, class_weight=class_weight_dict)

    # Optionnel : save_training_curves(history, model_names[i])
    pred_probs = model.predict(test_images, batch_size=BATCH_SIZE)
    predictions.append(pred_probs)

# --- Moyenne des prédictions (Ensemble) ---
ensemble_predictions = np.mean(predictions, axis=0)
ensemble_predicted_labels = np.argmax(ensemble_predictions, axis=1)
decoded_pred_labels = label_encoder.inverse_transform(ensemble_predicted_labels)

# --- Évaluation finale ---
print("\n📊 Rapport de classification (Ensemble) :")
print(classification_report(decoded_true_labels, decoded_pred_labels, target_names=class_names, zero_division=1))

# --- Scores globaux ---
f1 = f1_score(true_labels, ensemble_predicted_labels, average='weighted', zero_division=1)
precision = precision_score(true_labels, ensemble_predicted_labels, average='weighted', zero_division=1)
recall = recall_score(true_labels, ensemble_predicted_labels, average='weighted', zero_division=1)
accuracy = accuracy_score(true_labels, ensemble_predicted_labels)

print("\n🎯 Scores globaux (Ensemble) :")
print(f"Accuracy  : {accuracy:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")

# --- Matrice de confusion ---
def plot_confusion_matrix(y_true, y_pred, labels, model_name="Model"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f"Matrice de confusion : {model_name}")
    plt.xlabel("Prédictions")
    plt.ylabel("Vérités")
    plt.tight_layout()
    plt.savefig(f"conf_matrix_{model_name}.png")
    plt.show()

plot_confusion_matrix(decoded_true_labels, decoded_pred_labels, labels=class_names, model_name="Ensemble")
