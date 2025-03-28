# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 12:52:20 2025

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
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, BatchNormalization, MaxPooling2D, 
                                     Dropout, GlobalAveragePooling2D, Dense, 
                                     Concatenate, Add, Activation)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

#############################################
# 1. Încărcare și segmentare audio
#############################################

def extract_timestamp(filename):
    """Extrage timestamp-ul din numele fișierului de forma 'pva_..._YYYY-MM-DD-HHMMSS'."""
    match = re.search(r'pva_\d+_(\d{4})-(\d{2})-(\d{2})-(\d{2})(\d{2})(\d{2})', filename)
    if match:
        return f"{match.group(1)}/{match.group(2)}/{match.group(3)} {match.group(4)}:{match.group(5)}:{match.group(6)} UTC +0000"
    return None

def load_audio_and_segment(folder_path, csv_path):
    """
    Încarcă fișierele .wav din folder, extrage segmentul de interes (dacă există)
    și returnează o listă de dict-uri {'audio': vector_audio, 'sr': sample_rate, 'label': pdrs_score}.
    Se folosește o rată de eșantionare de 8000 Hz.
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
            if metadata_row['voice_code'].values[0] == 'bad':
                continue
            y, sr = librosa.load(file_path, sr=8000)
            voice_indexstart = metadata_row['voice_indexstart'].values[0]
            voice_indexend = metadata_row['voice_indexend'].values[0]
            if pd.notna(voice_indexstart) and pd.notna(voice_indexend):
                y = y[int(voice_indexstart):int(voice_indexend)]
            pdrs_score = metadata_row['pdrs_score'].values[0]
            samples.append({'audio': y, 'sr': sr, 'label': pdrs_score})
    return samples

#############################################
# 2. Funcții de augmentare (opțional, aici se poate adăuga augmentare suplimentară dacă dorești)
#############################################

def augment_audio_simple(y, sr):
    """
    Aplica augmentări subtile:
      - Time stretching (0.95-1.05, p=0.5)
      - Pitch shifting discret (±0.1 semitonuri, p=0.3)
    """
    if random.random() < 0.5:
        rate = random.uniform(0.95, 1.05)
        y = librosa.effects.time_stretch(y, rate=rate)
    if random.random() < 0.3:
        n_steps = random.uniform(-0.1, 0.1)
        y = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)
    return y

def create_augmented_dataset(samples, augment_factor=1):
    """
    Pentru fiecare eșantion, adaugă varianta originală și 'augment_factor' versiuni augmentate.
    """
    augmented_samples = []
    for sample in samples:
        y_original, sr, label = sample['audio'], sample['sr'], sample['label']
        augmented_samples.append({'audio': y_original, 'sr': sr, 'label': label})
        for _ in range(augment_factor):
            y_aug = augment_audio_simple(y_original, sr)
            augmented_samples.append({'audio': y_aug, 'sr': sr, 'label': label})
    return augmented_samples

#############################################
# 3. Extracția caracteristicilor din semnalul audio
#############################################

def audio_to_melspectrogram(y, sr, n_mels=128, fmax=4000, n_fft=512):
    """
    Transformă vectorul audio într-o spectrogramă Mel (în dB).
    Dacă semnalul este prea scurt, se completează cu zerouri până la n_fft.
    """
    if len(y) < n_fft:
        y = np.pad(y, (0, n_fft - len(y)), mode='constant')
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, n_mels=n_mels, fmax=fmax)
    return librosa.power_to_db(S, ref=np.max)

def get_voice_features(y, sr):
    """
    Extrage un set extins de caracteristici din semnalul audio:
      - 13 MFCC (medii)
      - spectral centroid, bandwidth, rolloff, zero crossing rate (medii)
      - spectral contrast (7 valori mediate, cu fmin=50)
      - tonnetz (6 valori mediate) (folosind chroma_cqt cu n_octaves=6)
      - RMS (medie)
      - Caracteristici suplimentare din pitch: media f0, deviația standard f0 și fracția de cadre voicing
    Toate valorile sunt prelucrate cu np.nan_to_num pentru a înlocui eventualele NaN.
    Returnează un vector de caracteristici.
    """
    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.nan_to_num(np.mean(mfcc, axis=1))
    # Spectral features
    spec_centroid = np.nan_to_num(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    spec_bandwidth = np.nan_to_num(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    rolloff = np.nan_to_num(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    zcr = np.nan_to_num(np.mean(librosa.feature.zero_crossing_rate(y=y)))
    # Spectral contrast
    stft = np.abs(librosa.stft(y=y))
    spec_contrast = librosa.feature.spectral_contrast(S=stft, sr=sr, fmin=50)
    spec_contrast_mean = np.nan_to_num(np.mean(spec_contrast, axis=1))
    # Tonnetz
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, n_octaves=6)
    tonnetz = librosa.feature.tonnetz(chroma=chroma)
    tonnetz_mean = np.nan_to_num(np.mean(tonnetz, axis=1))
    # RMS
    rms = np.nan_to_num(np.mean(librosa.feature.rms(y=y)))
    # Caracteristici din pitch (f0)
    try:
        f0, voiced_flag, _ = librosa.pyin(y, fmin=50, fmax=300)
    except Exception:
        f0 = np.array([np.nan])
        voiced_flag = np.array([0])
    f0 = np.nan_to_num(f0)  # înlocuiește NaN cu 0
    # Calculăm doar pentru cadrele voicing
    if np.sum(voiced_flag) > 0:
        mean_f0 = np.mean(f0[~np.isnan(f0)])
        std_f0 = np.std(f0[~np.isnan(f0)])
    else:
        mean_f0 = 0.0
        std_f0 = 0.0
    voiced_fraction = np.sum(~np.isnan(f0)) / (len(f0) + 1e-8)
    pitch_features = np.array([mean_f0, std_f0, voiced_fraction])
    pitch_features = np.nan_to_num(pitch_features)
    # Concatenăm toate caracteristicile (dimensiune totală: 13 + 4 + 7 + 6 + 1 + 3 = 34)
    features = np.concatenate([
        mfcc_mean,
        [spec_centroid, spec_bandwidth, rolloff, zcr],
        spec_contrast_mean,
        tonnetz_mean,
        [rms],
        pitch_features
    ])
    return features

def prepare_data(samples, fixed_width=200):
    """
    Pentru fiecare eșantion:
      - Calculează spectrograma Mel (dimensiune: 128 x fixed_width)
      - Extrage vectorul extins de caracteristici vocale (34 de valori)
    Returnează X_spec (spectrograme), X_feat (feature vector) și y (etichetă).
    """
    X_spec_list = []
    X_feat_list = []
    y_list = []
    for s in samples:
        y_audio, sr, label = s['audio'], s['sr'], s['label']
        spec = audio_to_melspectrogram(y_audio, sr)
        if spec.shape[1] < fixed_width:
            spec_fixed = np.pad(spec, ((0, 0), (0, fixed_width - spec.shape[1])), mode='constant')
        else:
            spec_fixed = spec[:, :fixed_width]
        X_spec_list.append(spec_fixed)
        X_feat_list.append(get_voice_features(y_audio, sr))
        y_list.append(label)
    X_spec = np.array(X_spec_list)[..., np.newaxis]  # (num_samples, 128, fixed_width, 1)
    X_feat = np.array(X_feat_list)  # (num_samples, 34)
    return X_spec, X_feat, np.array(y_list)

#############################################
# 4. Încărcare și pregătire date
#############################################

# Specifică căile către date
train_folder = '/home/administrator/Documents/Voce Parkinson/date/wav'
train_csv_path = '/home/administrator/Documents/Voce Parkinson/date/plmpva_train.csv'
test_folder = '/home/administrator/Documents/Voce Parkinson/date/wav test'
test_csv_path = '/home/administrator/Documents/Voce Parkinson/date/plmpva_test-WithPDRS.csv'

train_samples = load_audio_and_segment(train_folder, train_csv_path)
test_samples = load_audio_and_segment(test_folder, test_csv_path)
combined_samples = train_samples + test_samples

# Aplică augmentare (poți seta augment_factor dacă dorești)
augment_factor = 1
augmented_samples = create_augmented_dataset(combined_samples, augment_factor=augment_factor)

# Pregătește datele (spectrograme și feature vector extins)
X_spec, X_feat, y = prepare_data(augmented_samples, fixed_width=200)

# Normalizare spectrograme la [0,1]
X_spec_norm = (X_spec - X_spec.min()) / (X_spec.max() - X_spec.min())
# Standardizare pentru feature vector (folosim standard scaling)
X_feat_norm = (X_feat - X_feat.mean(axis=0)) / (X_feat.std(axis=0) + 1e-8)
# Scalare etichete (presupunem scoruri între 0 și 55)
y_norm = y / 55.0

# Împărțire train/val
X_spec_train, X_spec_val, X_feat_train, X_feat_val, y_train, y_val = train_test_split(
    X_spec_norm, X_feat_norm, y_norm, test_size=0.2, random_state=42
)

#############################################
# 5. Oversampling (bin-based) pentru setul de antrenament
#############################################

def oversample_regression(X1, X2, y, bins=10):
    df = pd.DataFrame({'y': y})
    df['bin'] = pd.cut(df['y'], bins=bins, labels=False)
    max_count = df['bin'].value_counts().max()
    X1_os, X2_os, y_os = [], [], []
    for b in df['bin'].unique():
        indices = df.index[df['bin'] == b].tolist()
        n_needed = max_count - len(indices)
        X1_os.extend(X1[indices])
        X2_os.extend(X2[indices])
        y_os.extend(y[indices])
        if n_needed > 0:
            rep_indices = np.random.choice(indices, n_needed, replace=True)
            X1_os.extend(X1[rep_indices])
            X2_os.extend(X2[rep_indices])
            y_os.extend(y[rep_indices])
    return np.array(X1_os), np.array(X2_os), np.array(y_os)

X_spec_train_os, X_feat_train_os, y_train_os = oversample_regression(X_spec_train, X_feat_train, y_train, bins=10)
print("Dimensiuni după oversampling:", X_spec_train_os.shape, X_feat_train_os.shape, y_train_os.shape)

#############################################
# 6. Model multi-input cu ResNet pentru spectrogramă
#############################################

def build_resnet_branch(input_tensor):
    """Construiește o ramură ResNet pentru procesarea spectrogramei."""
    def res_block(x, filters, kernel_size=(3,3), strides=(1,1)):
        shortcut = x
        x = Conv2D(filters, kernel_size, padding='same', strides=strides, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(filters, kernel_size, padding='same', strides=(1,1), activation=None)(x)
        x = BatchNormalization()(x)
        if shortcut.shape[-1] != filters or strides != (1,1):
            shortcut = Conv2D(filters, (1,1), padding='same', strides=strides, activation=None)(shortcut)
            shortcut = BatchNormalization()(shortcut)
        x = Add()([x, shortcut])
        x = Activation('relu')(x)
        return x

    x = Conv2D(32, (3,3), padding='same', activation='relu')(input_tensor)
    x = BatchNormalization()(x)
    x = res_block(x, 32)
    x = MaxPooling2D(pool_size=(2,1))(x)

    x = res_block(x, 64)
    x = MaxPooling2D(pool_size=(2,1))(x)

    x = res_block(x, 128)
    x = MaxPooling2D(pool_size=(2,1))(x)

    x = GlobalAveragePooling2D()(x)
    branch = Dense(64, activation='relu')(x)
    return branch

input_spec = Input(shape=X_spec_train_os.shape[1:])  # (128, 200, 1)
branch_spec = build_resnet_branch(input_spec)

# Ramura pentru vectorul extins de caracteristici vocale
input_feat = Input(shape=(X_feat_train_os.shape[1],))  # (34,)
branch_feat = Dense(32, activation='relu')(input_feat)
branch_feat = Dropout(0.2)(branch_feat)
branch_feat = Dense(32, activation='relu')(branch_feat)

# Combinare
combined = Concatenate()([branch_spec, branch_feat])
combined = Dense(64, activation='relu')(combined)
combined = Dropout(0.5)(combined)
output = Dense(1, activation='linear')(combined)

model = Model(inputs=[input_spec, input_feat], outputs=output)
optimizer = Adam(learning_rate=0.0003)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
model.summary()

#############################################
# 7. Antrenare
#############################################

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1)

history = model.fit(
    [X_spec_train_os, X_feat_train_os], y_train_os,
    validation_data=([X_spec_val, X_feat_val], y_val),
    epochs=100,
    batch_size=16,
    callbacks=[early_stop, lr_reduce]
)

#############################################
# 8. Evaluare și plot predicții vs. valori reale
#############################################

y_pred_norm = model.predict([X_spec_val, X_feat_val])
y_pred = y_pred_norm * 55.0
y_true = y_val * 55.0

mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print("RMSE: {:.4f}".format(rmse))
print("R2: {:.4f}".format(r2))
print("MAE: {:.4f}".format(mae))

plt.figure(figsize=(8,6))
plt.scatter(y_true, y_pred, alpha=0.6)
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
plt.xlabel("Valori reale PDRS")
plt.ylabel("Valori prezise PDRS")
plt.title("Predicții vs. Valori Reale (batch_size=64)")
plt.grid(True)
plt.show()
