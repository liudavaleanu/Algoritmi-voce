# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 16:21:50 2025

@author: Liuda
"""

import os
import re
import random
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error,
                             explained_variance_score, median_absolute_error, max_error)

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, BatchNormalization, MaxPooling2D, 
                                     Dropout, GlobalAveragePooling2D, Dense, 
                                     Concatenate, Add, Activation)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

##############################################################################
# 1. Încărcare și segmentare audio
##############################################################################

def extract_timestamp(filename):
    """Extrage timestamp-ul din numele fișierului de forma 'pva_..._YYYY-MM-DD-HHMMSS'."""
    match = re.search(r'pva_\d+_(\d{4})-(\d{2})-(\d{2})-(\d{2})(\d{2})(\d{2})', filename)
    if match:
        return f"{match.group(1)}/{match.group(2)}/{match.group(3)} {match.group(4)}:{match.group(5)}:{match.group(6)} UTC +0000"
    return None

def load_audio_and_segment(folder_path, csv_path):
    """
    Încarcă fișierele .wav din folder, extrage segmentul de interes (dacă există)
    și returnează o listă de dict-uri:
      [{'audio': <np.array>, 'sr': 8000, 'label': <pdrs_score>}, ...]
    """
    audio_metadata = pd.read_csv(csv_path)
    samples = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.wav'):
            file_path = os.path.join(folder_path, filename)
            call_timestamp = extract_timestamp(filename)
            metadata_row = audio_metadata[audio_metadata['call_timestamp'] == call_timestamp]
            if metadata_row.empty:
                continue
            
            # Excludem fișierele cu voice_code = 'bad'
            if metadata_row['voice_code'].values[0] == 'bad':
                continue
            
            # Încărcare audio la 8000 Hz
            y, sr = librosa.load(file_path, sr=8000)
            
            voice_indexstart = metadata_row['voice_indexstart'].values[0]
            voice_indexend = metadata_row['voice_indexend'].values[0]
            if pd.notna(voice_indexstart) and pd.notna(voice_indexend):
                y = y[int(voice_indexstart):int(voice_indexend)]
            
            pdrs_score = metadata_row['pdrs_score'].values[0]
            samples.append({'audio': y, 'sr': sr, 'label': pdrs_score})
    
    return samples

##############################################################################
# 2. Augmentare audio simplă (opțional)
##############################################################################

def augment_audio_simple(y, sr):
    """
    - Time stretching (0.95-1.05, p=0.5)
    - Pitch shifting (±0.1 semitonuri, p=0.3)
    """
    # Time stretching
    if random.random() < 0.5:
        rate = random.uniform(0.95, 1.05)
        y = librosa.effects.time_stretch(y, rate=rate)
    # Pitch shifting
    if random.random() < 0.3:
        n_steps = random.uniform(-0.1, 0.1)
        y = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)
    return y

def create_augmented_dataset(samples, augment_factor=1):
    """
    Pentru fiecare eșantion, adaugă varianta originală și 'augment_factor' copii augmentate.
    """
    augmented_samples = []
    for sample in samples:
        y_original = sample['audio']
        sr = sample['sr']
        label = sample['label']
        
        # Varianta originală
        augmented_samples.append({'audio': y_original, 'sr': sr, 'label': label})
        
        # Variante augmentate
        for _ in range(augment_factor):
            y_aug = augment_audio_simple(y_original, sr)
            augmented_samples.append({'audio': y_aug, 'sr': sr, 'label': label})
    
    return augmented_samples

##############################################################################
# 3. SpecAugment pe spectrogramă (mascare frecvență/timp)
##############################################################################

def spec_augment(spec, num_freq_masks=1, freq_mask_param=10,
                 num_time_masks=1, time_mask_param=15):
    """
    spec: (128, time)
    Mascare benzi pe frecvență și timp direct pe spectrogramă
    """
    augmented_spec = spec.copy()
    num_mel_channels = augmented_spec.shape[0]
    num_time_steps = augmented_spec.shape[1]
    
    # Mascare frecvență
    for _ in range(num_freq_masks):
        f = random.randint(0, freq_mask_param)
        f0 = random.randint(0, num_mel_channels - f)
        augmented_spec[f0:f0+f, :] = 0
    
    # Mascare timp
    for _ in range(num_time_masks):
        t = random.randint(0, time_mask_param)
        t0 = random.randint(0, num_time_steps - t)
        augmented_spec[:, t0:t0+t] = 0
    
    return augmented_spec

##############################################################################
# 4. Extracția spectrogramelor + feature engineering + SpecAugment
##############################################################################

def audio_to_melspectrogram(y, sr, n_mels=128, fmax=4000, n_fft=512):
    if len(y) < n_fft:
        y = np.pad(y, (0, n_fft - len(y)), mode='constant')
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, n_mels=n_mels, fmax=fmax)
    return librosa.power_to_db(S, ref=np.max)

def get_voice_features(y, sr):
    """
    Extrage un set extins de caracteristici (MFCC, spectral, pitch, etc.)
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.nan_to_num(np.mean(mfcc, axis=1))
    spec_centroid = np.nan_to_num(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    spec_bandwidth = np.nan_to_num(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    rolloff = np.nan_to_num(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    zcr = np.nan_to_num(np.mean(librosa.feature.zero_crossing_rate(y=y)))
    
    stft = np.abs(librosa.stft(y=y))
    spec_contrast = librosa.feature.spectral_contrast(S=stft, sr=sr, fmin=50)
    spec_contrast_mean = np.nan_to_num(np.mean(spec_contrast, axis=1))
    
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, n_octaves=6)
    tonnetz = librosa.feature.tonnetz(chroma=chroma)
    tonnetz_mean = np.nan_to_num(np.mean(tonnetz, axis=1))
    
    rms = np.nan_to_num(np.mean(librosa.feature.rms(y=y)))
    
    # pitch
    try:
        f0, voiced_flag, _ = librosa.pyin(y, fmin=50, fmax=300)
    except Exception:
        f0 = np.array([np.nan])
        voiced_flag = np.array([False])
    f0 = np.nan_to_num(f0)
    if np.sum(voiced_flag) > 0:
        mean_f0 = np.mean(f0[f0 > 0])
        std_f0 = np.std(f0[f0 > 0])
    else:
        mean_f0 = 0.0
        std_f0 = 0.0
    voiced_fraction = np.sum(f0 > 0) / (len(f0) + 1e-8)
    
    pitch_features = np.array([mean_f0, std_f0, voiced_fraction])
    pitch_features = np.nan_to_num(pitch_features)
    
    features = np.concatenate([
        mfcc_mean,
        [spec_centroid, spec_bandwidth, rolloff, zcr],
        spec_contrast_mean,
        tonnetz_mean,
        [rms],
        pitch_features
    ])
    return features

def prepare_data(samples, fixed_width=200,
                 apply_specaugment=False,
                 num_freq_masks=1, freq_mask_param=10,
                 num_time_masks=1, time_mask_param=15):
    """
    - Convertește semnalul audio la spectrogramă
    - (Opțional) aplică SpecAugment
    - Redimensionează la (128, fixed_width)
    - Extrage feature-uri suplimentare
    """
    X_spec_list = []
    X_feat_list = []
    y_list = []
    
    for s in samples:
        y_audio = s['audio']
        sr = s['sr']
        label = s['label']
        
        spec = audio_to_melspectrogram(y_audio, sr)
        
        # Aplica SpecAugment DOAR dacă e setat (ex. doar pentru train)
        if apply_specaugment:
            spec = spec_augment(spec,
                                num_freq_masks=num_freq_masks,
                                freq_mask_param=freq_mask_param,
                                num_time_masks=num_time_masks,
                                time_mask_param=time_mask_param)
        
        # Redimensionare la fixed_width
        if spec.shape[1] < fixed_width:
            pad_width = fixed_width - spec.shape[1]
            spec_fixed = np.pad(spec, ((0,0),(0,pad_width)), mode='constant')
        else:
            spec_fixed = spec[:, :fixed_width]
        
        X_spec_list.append(spec_fixed)
        X_feat_list.append(get_voice_features(y_audio, sr))
        y_list.append(label)
    
    X_spec = np.array(X_spec_list)[..., np.newaxis]  # (n, 128, fixed_width, 1)
    X_feat = np.array(X_feat_list)                  # (n, nr_features)
    y_arr = np.array(y_list)
    
    return X_spec, X_feat, y_arr

##############################################################################
# 5. Oversampling direcționat pentru scoruri mari
##############################################################################

def oversample_high_scores(X_spec, X_feat, y, threshold=40.0, factor=2):
    """
    Replică de 'factor' ori eșantioanele cu scor > threshold.
    """
    X_spec_list, X_feat_list, y_list = [], [], []
    
    for i in range(len(y)):
        X_spec_list.append(X_spec[i])
        X_feat_list.append(X_feat[i])
        y_list.append(y[i])
        
        if y[i] > threshold:
            # replicăm factor-1 ori
            for _ in range(factor-1):
                X_spec_list.append(X_spec[i])
                X_feat_list.append(X_feat[i])
                y_list.append(y[i])
    
    return (np.array(X_spec_list),
            np.array(X_feat_list),
            np.array(y_list))

##############################################################################
# 6. Model ResNet + Dense (multi-input)
##############################################################################

def build_resnet_branch(input_tensor):
    """Ramură ResNet cu L2 pe Conv2D."""
    l2_reg = regularizers.l2(1e-5)
    
    def res_block(x, filters, kernel_size=(3,3), strides=(1,1)):
        shortcut = x
        x = Conv2D(filters, kernel_size, padding='same', strides=strides, 
                   activation='relu', kernel_regularizer=l2_reg)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.1)(x)  # Dropout suplimentar în blocul rezidual
        x = Conv2D(filters, kernel_size, padding='same', strides=(1,1), 
                   activation=None, kernel_regularizer=l2_reg)(x)
        x = BatchNormalization()(x)
        if shortcut.shape[-1] != filters or strides != (1,1):
            shortcut = Conv2D(filters, (1,1), padding='same', strides=strides, 
                              activation=None, kernel_regularizer=l2_reg)(shortcut)
            shortcut = BatchNormalization()(shortcut)
        x = Add()([x, shortcut])
        x = Activation('relu')(x)
        return x

    # Primul conv
    x = Conv2D(32, (3,3), padding='same', activation='relu', kernel_regularizer=l2_reg)(input_tensor)
    x = BatchNormalization()(x)
    
    # Blocuri reziduale + pooling
    x = res_block(x, 32)
    x = MaxPooling2D(pool_size=(2,1))(x)

    x = res_block(x, 64)
    x = MaxPooling2D(pool_size=(2,1))(x)

    x = res_block(x, 128)
    x = MaxPooling2D(pool_size=(2,1))(x)

    x = GlobalAveragePooling2D()(x)
    
    # Strat final conv
    branch = Dense(64, activation='relu', kernel_regularizer=l2_reg)(x)
    return branch

def build_model_resnet(input_shape_spec, input_shape_feat, lr=0.0005):
    l2_reg = regularizers.l2(1e-5)
    
    input_spec = Input(shape=input_shape_spec)  # (128, 200, 1)
    branch_spec = build_resnet_branch(input_spec)

    input_feat = Input(shape=(input_shape_feat,))
    branch_feat = Dense(32, activation='relu', kernel_regularizer=l2_reg)(input_feat)
    branch_feat = Dropout(0.2)(branch_feat)
    branch_feat = Dense(32, activation='relu', kernel_regularizer=l2_reg)(branch_feat)

    combined = Concatenate()([branch_spec, branch_feat])
    combined = Dense(64, activation='relu', kernel_regularizer=l2_reg)(combined)
    combined = Dropout(0.6)(combined)
    output = Dense(1, activation='linear', kernel_regularizer=l2_reg)(combined)

    model = Model(inputs=[input_spec, input_feat], outputs=output)
    optimizer = Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
    return model

##############################################################################
# 7. Integrare completă (Exemplu flux)
##############################################################################

# 1. Încarcă datele
train_folder = r'D:\Liuda\proiect cu PVA\wav'
train_csv_path = r'D:\Liuda\proiect cu PVA\plmpva_train.csv'
test_folder = r'D:\Liuda\proiect cu PVA\wav test'
test_csv_path = r'D:\Liuda\proiect cu PVA\plmpva_test-WithPDRS.csv'

train_samples = load_audio_and_segment(train_folder, train_csv_path)
test_samples = load_audio_and_segment(test_folder, test_csv_path)
combined_samples = train_samples + test_samples

# 2. Augmentare audio (opțional) - setăm un factor scăzut
augment_factor = 1
augmented_samples = create_augmented_dataset(combined_samples, augment_factor=augment_factor)

# 3. Împărțim (ex. direct train/val) - 
#    Poți face split înainte sau după, depinde de fluxul tău. 
#    De exemplu, folosim direct splitted in train/val:
#    Dar aici facem TOTUL intr-un singur set + oversample -> 
#    Ideal e să ții un set de validare separat.

# 4. Aplica prepare_data PE SETUL DE ANTRENAMENT cu SpecAugment, 
#    iar pe setul de validare/test FĂRĂ SpecAugment
#    => In acest exemplu, transformăm TOTUL cu SpecAugment, dar 
#       e indicat să separi train vs. val/test clar.

apply_specaug = True  # Doar exemplu
X_spec, X_feat, y = prepare_data(
    samples=augmented_samples,
    fixed_width=200,
    apply_specaugment=apply_specaug,
    num_freq_masks=1, freq_mask_param=10,
    num_time_masks=1, time_mask_param=15
)

# 5. Normalizare spectrogramă
X_spec_min, X_spec_max = X_spec.min(), X_spec.max()
X_spec_norm = (X_spec - X_spec_min) / (X_spec_max - X_spec_min)

# 6. Standardizare features
X_feat_mean = X_feat.mean(axis=0)
X_feat_std = X_feat.std(axis=0) + 1e-8
X_feat_norm = (X_feat - X_feat_mean) / X_feat_std

# 7. Scalare etichete (dacă max e 55)
y_max = 55.0
y_norm = y / y_max

# 8. Împărțire train/val
X_spec_train, X_spec_val, X_feat_train, X_feat_val, y_train, y_val = train_test_split(
    X_spec_norm, X_feat_norm, y_norm, test_size=0.2, random_state=42
)

# 9. Oversampling direcționat (pentru scorurile mari > 40, factor=2)
threshold_score = 40.0
factor_score = 2
X_spec_train_os, X_feat_train_os, y_train_os = oversample_high_scores(
    X_spec_train, X_feat_train, y_train, threshold=threshold_score, factor=factor_score
)

print("Dimensiuni după oversampling direcționat:", X_spec_train_os.shape, X_feat_train_os.shape, y_train_os.shape)

# 10. Construim modelul
model = build_model_resnet(
    input_shape_spec=X_spec_train_os.shape[1:],
    input_shape_feat=X_feat_train_os.shape[1],
    lr=0.0005
)

model.summary()

# 11. Antrenare
early_stop = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)
lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

history = model.fit(
    [X_spec_train_os, X_feat_train_os], y_train_os,
    validation_data=([X_spec_val, X_feat_val], y_val),
    epochs=80,
    batch_size=32,
    callbacks=[early_stop, lr_reduce]
)

# 12. Evaluare
y_pred_norm = model.predict([X_spec_val, X_feat_val])
y_pred = y_pred_norm * y_max
y_true = y_val * y_max

mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
exp_var = explained_variance_score(y_true, y_pred)
median_err = median_absolute_error(y_true, y_pred)
max_err = max_error(y_true, y_pred)

print(f"RMSE: {rmse:.4f}")
print(f"R2: {r2:.4f}")
print(f"MAE: {mae:.4f}")
print(f"Explained Variance: {exp_var:.4f}")
print(f"Median Absolute Error: {median_err:.4f}")
print(f"Max Error: {max_err:.4f}")

# Plot antrenament
def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # Loss
    axes[0].plot(history.history['loss'], label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_title('Evoluția Loss-ului')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].legend()
    # MAE
    axes[1].plot(history.history['mae'], label='Train MAE')
    axes[1].plot(history.history['val_mae'], label='Val MAE')
    axes[1].set_title('Evoluția MAE')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].legend()
    plt.show()

plot_training_history(history)

# Plot predicții vs. real
plt.figure(figsize=(6,6))
plt.scatter(y_true, y_pred, alpha=0.6)
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
plt.xlabel("Valori Reale PDRS")
plt.ylabel("Valori Prezise PDRS")
plt.title("Predicții vs. Valori Reale")
plt.grid(True)
plt.show()


