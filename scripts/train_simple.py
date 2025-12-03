#!/usr/bin/env python3
"""
🌸 Plant Classifier - Basit Eğitim Scripti
Flowers102 + MobileNetV2
"""

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from pathlib import Path
import sys

print("=" * 50)
print("🌸 Plant Classifier Eğitimi")
print("=" * 50)

print("\n📥 Dataset indiriliyor...")
(train_ds, val_ds, test_ds), metadata = tfds.load(
    'oxford_flowers102',
    split=['train', 'validation', 'test'],
    with_info=True,
    as_supervised=True
)

NUM_CLASSES = metadata.features['label'].num_classes
print(f"✅ Dataset yüklendi: {NUM_CLASSES} sınıf")
print(f"   - Eğitim: {len(train_ds)} örnek")
print(f"   - Validation: {len(val_ds)} örnek")
print(f"   - Test: {len(test_ds)} örnek")

print("\n🔄 Veri işleniyor...")
IMG_SIZE = 224
BATCH_SIZE = 32

def preprocess(image, label):
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image, label

def augment(image, label):
    image = tf.image.resize(image, [IMG_SIZE + 20, IMG_SIZE + 20])
    image = tf.image.random_crop(image, [IMG_SIZE, IMG_SIZE, 3])
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.2)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image, label

train_ds = train_ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_ds = val_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_ds = test_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

print("✅ Dataset hazır!")

print("\n🧠 Model oluşturuluyor...")
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("✅ Model hazır!")
print(f"   - Toplam parametre: {model.count_params():,}")

print("\n🚀 Eğitim başlıyor...")
print("=" * 50)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3
    )
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=callbacks,
    verbose=1
)

print("\n📊 Test ediliyor...")
test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)
print(f"✅ Test Accuracy: {test_accuracy * 100:.2f}%")


print("\n💾 Model kaydediliyor...")
output_dir = Path(__file__).parent.parent / 'output'
output_dir.mkdir(exist_ok=True)

model_path = output_dir / 'plant_classifier.keras'
model.save(model_path)
print(f"✅ Model kaydedildi: {model_path}")

# CoreML'e dönüştür
print("\n🔄 CoreML'e dönüştürülüyor...")
try:
    import coremltools as ct
    
    # Label isimlerini yükle
    labels_path = Path(__file__).parent.parent / 'labels.txt'
    with open(labels_path, 'r') as f:
        class_labels = [line.strip() for line in f.readlines() if line.strip()]
    
    print(f"   - {len(class_labels)} sınıf label'ı yüklendi")
    
    # Model input ismini al
    input_name = model.input.name.split(':')[0]
    print(f"   - Model input ismi: {input_name}")
    
    # CoreML'e dönüştür
    coreml_model = ct.convert(
        model,
        inputs=[
            ct.ImageType(
                name=input_name,
                shape=(1, 224, 224, 3),
                scale=1/127.5,  # MobileNetV2 preprocessing
                bias=[-1, -1, -1],
                color_layout=ct.colorlayout.RGB
            )
        ],
        classifier_config=ct.ClassifierConfig(class_labels)
    )
    
    # Metadata ekle
    coreml_model.author = "Plant Classifier"
    coreml_model.short_description = "Flowers102 dataset ile eğitilmiş çiçek sınıflandırıcı (102 sınıf)"
    coreml_model.version = "1.0"
    
    # Kaydet
    coreml_path = output_dir / 'PlantClassifier.mlpackage'
    coreml_model.save(str(coreml_path))
    print(f"✅ CoreML model kaydedildi: {coreml_path}")
    
except Exception as e:
    print(f"⚠️  CoreML dönüşümü başarısız: {e}")
    print("   Keras model kaydedildi, CoreML'e manuel dönüştürebilirsiniz.")
    print("   Manuel dönüştürme: python scripts/convert_to_coreml.py")

print("\n" + "=" * 50)
print("🎉 EĞİTİM TAMAMLANDI!")
print("=" * 50)
print(f"\nFinal Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Model: {model_path}")
