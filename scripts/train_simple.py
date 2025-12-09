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
    # Güçlü augmentation - %85 için gerekli
    image = tf.image.resize(image, [IMG_SIZE + 40, IMG_SIZE + 40])
    image = tf.image.random_crop(image, [IMG_SIZE, IMG_SIZE, 3])
    image = tf.image.random_flip_left_right(image)
    # Renk augmentasyonları
    image = tf.image.random_brightness(image, 0.3)
    image = tf.image.random_contrast(image, 0.7, 1.3)
    image = tf.image.random_saturation(image, 0.7, 1.3)
    image = tf.image.random_hue(image, 0.1)
    # Random erase (cutout benzeri)
    if tf.random.uniform([]) > 0.5:
        h, w = IMG_SIZE // 4, IMG_SIZE // 4
        y = tf.random.uniform([], 0, IMG_SIZE - h, dtype=tf.int32)
        x = tf.random.uniform([], 0, IMG_SIZE - w, dtype=tf.int32)
        mask = tf.ones([h, w, 3])
        mask = tf.pad(mask, [[y, IMG_SIZE - h - y], [x, IMG_SIZE - w - x], [0, 0]])
        image = image * (1 - mask) + mask * tf.random.uniform([1, 1, 3], -1, 1)
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
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(768, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),
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

# Cosine decay learning rate
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.001,
    decay_steps=30 * len(train_ds),
    alpha=0.0001
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        mode='max'
    )
]

print("📍 Aşama 1: Transfer Learning (üst katmanları eğit - 30 epoch)")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30,
    callbacks=callbacks,
    verbose=1
)

# Fine-tuning Aşama 1: Base model'in yarısını aç
print("\n📍 Aşama 2: Fine-tuning (base model'in yarısı - 20 epoch)")
base_model.trainable = True

# İlk yarı dondur, ikinci yarı eğit
for layer in base_model.layers[:len(base_model.layers)//2]:
    layer.trainable = False

print(f"   - Eğitilebilir base katman: {sum([1 for l in base_model.layers if l.trainable])}")

lr_schedule_fine1 = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.0001,
    decay_steps=20 * len(train_ds),
    alpha=0.00001
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule_fine1),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history_fine1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=callbacks,
    verbose=1
)

# Fine-tuning Aşama 2: Tüm base model'i aç
print("\n📍 Aşama 3: Full Fine-tuning (tüm model - 15 epoch)")
for layer in base_model.layers:
    layer.trainable = True

print(f"   - Eğitilebilir toplam katman: {len(model.trainable_variables)}")

lr_schedule_fine2 = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.00003,
    decay_steps=15 * len(train_ds),
    alpha=0.000001
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule_fine2),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history_fine2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
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