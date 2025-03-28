# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 14:40:36 2025

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

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, BatchNormalization, MaxPooling2D, 
                                     Dropout, GlobalAveragePooling2D, Dense, 
                                     Concatenate, Add, Activation)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

##############################################################################
# 1. Încărcare și segmentare audio
##############################################################################

def extract_timestamp(filename):
    match = re.search(r'pva_\d+_(\d{4})-(\d{2})-(\d{2})-(\d{2})(\d{2})(\d{2})', filename)
    if match:
        return f"{match.group(1)}/{match.group(2)}/{match.group(3)} {match.group(4)}:{match.group(5)}:{match.group(6)} UTC +0000"
    return None

def load_audio_and_segment(folder_path, csv_path):
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
            
            y, sr = librosa.load(file_path, sr=8000)
            voice_indexstart = metadata_row['voice_indexstart'].values[0]
            voice_indexend = metadata_row['voice_indexend'].values[0]
            if pd.notna(voice_indexstart) and pd.notna(voice_indexend):
                y = y[int(voice_indexstart):int(voice_indexend)]
            
            pdrs_score = metadata_row['pdrs_score'].values[0]
            samples.append({'audio': y, 'sr': sr, 'label': pdrs_score})
    return samples

##############################################################################
# 2. Augmentare audio (time stretching + pitch shifting)
##############################################################################

def augment_audio_simple(y, sr):
    """
    Time stretching (0.95-1.05, p=0.5)
    Pitch shifting (±0.1 semitonuri, p=0.3)
    """
    if random.random() < 0.5:
        rate = random.uniform(0.95, 1.05)
        y = librosa.effects.time_stretch(y, rate=rate)
    if random.random() < 0.3:
        n_steps = random.uniform(-0.1, 0.1)
        y = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)
    return y

def create_augmented_dataset(samples, augment_factor=1):
    augmented_samples = []
    for s in samples:
        y_original, sr, label = s['audio'], s['sr'], s['label']
        # Varianta originală
        augmented_samples.append({'audio': y_original, 'sr': sr, 'label': label})
        # Augmentare
        for _ in range(augment_factor):
            y_aug = augment_audio_simple(y_original, sr)
            augmented_samples.append({'audio': y_aug, 'sr': sr, 'label': label})
    return augmented_samples

##############################################################################
# 3. Conversie la spectrogramă + features
##############################################################################

def audio_to_melspectrogram(y, sr, n_mels=128, fmax=4000, n_fft=512):
    if len(y) < n_fft:
        y = np.pad(y, (0, n_fft - len(y)), mode='constant')
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=fmax, n_fft=n_fft)
    return librosa.power_to_db(S, ref=np.max)

def get_voice_features(y, sr):
    # 13 MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.nan_to_num(np.mean(mfcc, axis=1))
    # Spectral
    spec_centroid = np.nan_to_num(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    spec_bandwidth = np.nan_to_num(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    rolloff = np.nan_to_num(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    zcr = np.nan_to_num(np.mean(librosa.feature.zero_crossing_rate(y=y)))
    # Contrast
    stft = np.abs(librosa.stft(y=y))
    spec_contrast = librosa.feature.spectral_contrast(S=stft, sr=sr, fmin=50)
    spec_contrast_mean = np.nan_to_num(np.mean(spec_contrast, axis=1))
    # Tonnetz
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, n_octaves=6)
    tonnetz = librosa.feature.tonnetz(chroma=chroma)
    tonnetz_mean = np.nan_to_num(np.mean(tonnetz, axis=1))
    # RMS
    rms = np.nan_to_num(np.mean(librosa.feature.rms(y=y)))
    # Pitch
    try:
        f0, voiced_flag, _ = librosa.pyin(y, fmin=50, fmax=300)
    except:
        f0 = np.array([np.nan])
        voiced_flag = np.array([False])
    f0 = np.nan_to_num(f0)
    if np.sum(voiced_flag) > 0:
        mean_f0 = np.mean(f0[f0>0])
        std_f0 = np.std(f0[f0>0])
    else:
        mean_f0 = 0.0
        std_f0 = 0.0
    voiced_fraction = np.sum(f0>0)/(len(f0)+1e-8)
    
    pitch_features = np.array([mean_f0, std_f0, voiced_fraction])
    
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
    X_spec_list = []
    X_feat_list = []
    y_list = []
    for s in samples:
        y_audio, sr, label = s['audio'], s['sr'], s['label']
        spec = audio_to_melspectrogram(y_audio, sr)
        # Fix width
        if spec.shape[1] < fixed_width:
            pad_width = fixed_width - spec.shape[1]
            spec_fixed = np.pad(spec, ((0,0),(0,pad_width)), mode='constant')
        else:
            spec_fixed = spec[:, :fixed_width]
        
        X_spec_list.append(spec_fixed)
        X_feat_list.append(get_voice_features(y_audio, sr))
        y_list.append(label)
    
    X_spec = np.array(X_spec_list)[..., np.newaxis]  # (n, 128, fixed_width, 1)
    X_feat = np.array(X_feat_list)                  # (n, 34) approx
    y_arr = np.array(y_list)
    return X_spec, X_feat, y_arr

##############################################################################
# 4. Oversampling bin-based
##############################################################################

def oversample_regression(X_spec, X_feat, y, bins=10):
    """
    Metoda bin-based: se împarte y în 'bins' intervale și se replicate 
    eșantioanele din binurile mai puțin populate.
    """
    df = pd.DataFrame({'y': y})
    df['bin'] = pd.cut(df['y'], bins=bins, labels=False)
    counts = df['bin'].value_counts()
    max_count = counts.max()
    
    X_spec_os, X_feat_os, y_os = [], [], []
    for b in df['bin'].unique():
        indices = df.index[df['bin'] == b].tolist()
        n_needed = max_count - len(indices)
        # Adăugăm toate eșantioanele existente
        for idx in indices:
            X_spec_os.append(X_spec[idx])
            X_feat_os.append(X_feat[idx])
            y_os.append(y[idx])
        # Replicăm
        if n_needed > 0:
            rep_indices = np.random.choice(indices, n_needed, replace=True)
            for ridx in rep_indices:
                X_spec_os.append(X_spec[ridx])
                X_feat_os.append(X_feat[ridx])
                y_os.append(y[ridx])
    
    return (np.array(X_spec_os),
            np.array(X_feat_os),
            np.array(y_os))

##############################################################################
# 5. Weighted MSE (pentru scor <10 și >38)
##############################################################################

import tensorflow.keras.backend as K

def weighted_mse(y_true, y_pred):
    """
    Dacă scorul (y_true) < 10/55 ~ 0.18 sau > 38/55 ~ 0.69, 
    se penalizează erorile cu factor=2, altfel factor=1.
    """
    threshold_low = 10.0/55.0
    threshold_high = 38.0/55.0
    
    mask_low = K.cast(K.less(y_true, threshold_low), K.floatx())     # 1 dacă y_true < 0.18
    mask_high = K.cast(K.greater(y_true, threshold_high), K.floatx())# 1 dacă y_true > 0.69
    factor = 1.0 + mask_low + mask_high  # devine 2 dacă e doar una, 3 dacă ambele (teoretic imposibil)
    
    error = y_true - y_pred
    return K.mean(factor * K.square(error), axis=-1)

##############################################################################
# 6. Construim modelul ResNet + multi-input
##############################################################################

def build_resnet_branch(input_tensor):
    x = Conv2D(32, (3,3), padding='same', activation='relu')(input_tensor)
    x = BatchNormalization()(x)
    
    def res_block(x, filters):
        shortcut = x
        x = Conv2D(filters, (3,3), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(filters, (3,3), padding='same', activation=None)(x)
        x = BatchNormalization()(x)
        if shortcut.shape[-1] != filters:
            shortcut = Conv2D(filters, (1,1), padding='same')(shortcut)
            shortcut = BatchNormalization()(shortcut)
        x = Add()([x, shortcut])
        x = Activation('relu')(x)
        return x
    
    x = res_block(x, 32)
    x = MaxPooling2D(pool_size=(2,1))(x)
    x = res_block(x, 64)
    x = MaxPooling2D(pool_size=(2,1))(x)
    x = res_block(x, 128)
    x = MaxPooling2D(pool_size=(2,1))(x)
    
    x = GlobalAveragePooling2D()(x)
    branch = Dense(64, activation='relu')(x)
    return branch

def build_model(input_shape_spec, input_shape_feat, lr=0.0003):
    input_spec = Input(shape=input_shape_spec)
    branch_spec = build_resnet_branch(input_spec)
    
    input_feat = Input(shape=(input_shape_feat,))
    x = Dense(32, activation='relu')(input_feat)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu')(x)
    
    combined = Concatenate()([branch_spec, x])
    combined = Dense(64, activation='relu')(combined)
    combined = Dropout(0.5)(combined)
    output = Dense(1, activation='linear')(combined)
    
    model = Model(inputs=[input_spec, input_feat], outputs=output)
    opt = Adam(learning_rate=lr)
    # Observă: Weighted MSE + mae
    model.compile(optimizer=opt, loss=weighted_mse, metrics=['mae'])
    return model

##############################################################################
# 7. CROSS-VALIDATION + antrenament final + test
##############################################################################

def cross_validate_model(X_spec, X_feat, y, n_folds=5, epochs=25, batch_size=16, lr=0.0003):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    r2_scores = []
    
    fold_idx = 1
    for train_idx, val_idx in kf.split(X_spec):
        X_spec_train_fold, X_spec_val_fold = X_spec[train_idx], X_spec[val_idx]
        X_feat_train_fold, X_feat_val_fold = X_feat[train_idx], X_feat[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Oversampling bin-based
        X_spec_os, X_feat_os, y_os = oversample_regression(X_spec_train_fold, X_feat_train_fold, y_train_fold, bins=10)
        
        model_cv = build_model(
            input_shape_spec=X_spec_os.shape[1:],
            input_shape_feat=X_feat_os.shape[1],
            lr=lr
        )
        
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=0)
        
        model_cv.fit(
            [X_spec_os, X_feat_os], y_os,
            validation_data=([X_spec_val_fold, X_feat_val_fold], y_val_fold),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, lr_reduce],
            verbose=0
        )
        
        y_pred_val = model_cv.predict([X_spec_val_fold, X_feat_val_fold]) * 55.0
        y_val_real = y_val_fold * 55.0
        r2_fold = r2_score(y_val_real, y_pred_val)
        r2_scores.append(r2_fold)
        
        print(f"Fold {fold_idx}: R2 = {r2_fold:.4f}")
        fold_idx += 1
    
    return r2_scores

def main():
    # 1. Încarcă date (train + test) și amestecă
    train_folder = r'/home/administrator/Documents/Voce Parkinson/date/wav'
    train_csv_path = r'/home/administrator/Documents/Voce Parkinson/date/plmpva_train.csv'
    test_folder = r'/home/administrator/Documents/Voce Parkinson/date/wav test'
    test_csv_path = r'/home/administrator/Documents/Voce Parkinson/date/plmpva_test-WithPDRS.csv'
    
    train_samples = load_audio_and_segment(train_folder, train_csv_path)
    test_samples = load_audio_and_segment(test_folder, test_csv_path)
    combined_samples = train_samples + test_samples
    random.shuffle(combined_samples)
    
    # 2. Augmentare audio
    augment_factor = 1
    augmented_samples = create_augmented_dataset(combined_samples, augment_factor=augment_factor)
    
    # 3. Pregătește date
    X_spec, X_feat, y = prepare_data(augmented_samples, fixed_width=200)
    # Normalizare spectrograme
    X_spec_min, X_spec_max = X_spec.min(), X_spec.max()
    X_spec_norm = (X_spec - X_spec_min) / (X_spec_max - X_spec_min)
    # Standardizare features
    X_feat_mean = X_feat.mean(axis=0)
    X_feat_std = X_feat.std(axis=0) + 1e-8
    X_feat_norm = (X_feat - X_feat_mean) / X_feat_std
    # Scalare scoruri la 0..1 (max 55)
    y_norm = y / 55.0
    
    # 4. Separăm un set final de test
    X_spec_trainval, X_spec_test, X_feat_trainval, X_feat_test, y_trainval, y_test = train_test_split(
        X_spec_norm, X_feat_norm, y_norm, test_size=0.2, random_state=42
    )
    
    # 5. Cross-validation pe setul de antrenament (trainval)
    folds = 5
    epochs = 25
    batch_size = 16
    lr = 0.0003
    
    r2_scores = cross_validate_model(
        X_spec_trainval, X_feat_trainval, y_trainval,
        n_folds=folds, epochs=epochs, batch_size=batch_size, lr=lr
    )
    print("R2 pe fiecare fold:", r2_scores)
    print("Media R2:", np.mean(r2_scores))
    
    # Plot variabilitatea R2 pe fold-uri
    plt.figure(figsize=(6,4))
    plt.bar(range(1, folds+1), r2_scores, color='skyblue')
    plt.axhline(np.mean(r2_scores), color='r', linestyle='--', label='Media R2')
    plt.xlabel("Fold")
    plt.ylabel("R2")
    plt.title("Variabilitatea R2 pe Fold-uri (Cross-Validation)")
    plt.legend()
    plt.show()
    
    # 6. Antrenăm modelul final pe TOT setul de trainval + oversampling
    X_spec_trainval_os, X_feat_trainval_os, y_trainval_os = oversample_regression(X_spec_trainval, X_feat_trainval, y_trainval, bins=10)
    
    model_final = build_model(
        input_shape_spec=X_spec_trainval_os.shape[1:],
        input_shape_feat=X_feat_trainval_os.shape[1],
        lr=lr
    )
    
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    
    history = model_final.fit(
        [X_spec_trainval_os, X_feat_trainval_os], y_trainval_os,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, lr_reduce],
        verbose=1
    )
    
    # 7. Evaluare pe setul final de test
    y_pred_norm = model_final.predict([X_spec_test, X_feat_test])
    y_pred = y_pred_norm.flatten() * 55.0
    y_test_real = y_test * 55.0
    
    mse_test = mean_squared_error(y_test_real, y_pred)
    rmse_test = np.sqrt(mse_test)
    mae_test = mean_absolute_error(y_test_real, y_pred)
    r2_test = r2_score(y_test_real, y_pred)
    
    print(f"\nEvaluare finală pe setul de test:")
    print(f"RMSE={rmse_test:.2f}, R2={r2_test:.2f}, MAE={mae_test:.2f}")
    
    # Plot reziduurile pe setul de test
    residuals = y_test_real - y_pred
    plt.figure(figsize=(6,4))
    plt.hist(residuals, bins=30, alpha=0.7, color='purple')
    plt.title("Distribuția Reziduurilor (set de test)")
    plt.xlabel("Eroare (y_true - y_pred)")
    plt.ylabel("Număr eșantioane")
    plt.grid(True)
    plt.show()
    
    # Observa scorurile <10 și >38
    print("\nObservație: scorurile <10 și >38 sunt rare => predicții mai slabe probabil.\n")

if __name__ == "__main__":
    main()
