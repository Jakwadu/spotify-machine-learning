import os
import librosa
import warnings
import random
import tensorflow as tf
import numpy as np
import pickle
import pandas as pd
from tqdm import trange
from models import build_classifier
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

DATA_DIR = 'C:\\Users\\Jamie\\Documents\\MiscProjects\\spotify-machine-learning\\all_artists'
DEFAULT_VALIDATION_SPLIT = 0.3
DEFAULT_BATCH_SIZE = 80
SAMPLE_RATE = 22050
SNIPPET_LENGTH = 3
SAMPLE_LIMIT = SNIPPET_LENGTH*SAMPLE_RATE

np.random.seed(123)
tf.random.set_seed(123)
tf.keras.mixed_precision.set_global_policy('mixed_float16')


def generate_bulk_training_data(data_directory, validation_split=DEFAULT_VALIDATION_SPLIT, scale_data=True):
    artist_dirs = [os.path.join(data_directory, dir_) for dir_ in os.listdir(data_directory)]
    label_dict = {os.path.basename(artist_dir): idx for idx, artist_dir in enumerate(artist_dirs)}
    song_splits = []
    for idx in trange(len(artist_dirs), desc='Creating snippets from artist song samples'):
        artist = os.path.basename(artist_dirs[idx])
        song_files = [os.path.join(artist_dirs[idx], f) for f in os.listdir(artist_dirs[idx]) if '.mp3' in f]
        for file in song_files:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                audio_data, sr = librosa.load(file, sr=SAMPLE_RATE)
            audio_data = audio_data.astype(np.float16)
            n_splits = len(audio_data) // SAMPLE_LIMIT
            audio_data = audio_data[:n_splits * SAMPLE_LIMIT]
            audio_data = np.split(audio_data, list(range(SAMPLE_LIMIT, len(audio_data), SAMPLE_LIMIT)))
            audio_data = [[data, label_dict[artist]] for data in audio_data]
            song_splits += audio_data
    random.shuffle(song_splits)
    threshold = int(len(song_splits)*(1-validation_split))
    training = song_splits[:threshold]
    validation = song_splits[threshold:]
    x_train = np.array([split[0] for split in training])
    y_train = np.array([split[1] for split in training])
    x_val = np.array([split[0] for split in validation])
    y_val = np.array([split[1] for split in validation])

    if scale_data:
        max_ = max(x_train.max(), x_val.max())
        min_ = min(x_train.min(), x_val.min())
        range_ = max_ - min_
        print('****** Scaling Parameters ******')
        print(f'Max: {max_:.3f}')
        print(f'Min: {min_:.3f}')
        x_train = (x_train - min_) / range_
        x_val = (x_val - min_) / range_

    y_train = tf.keras.utils.to_categorical(y_train)
    y_val = tf.keras.utils.to_categorical(y_val)

    return (x_train, y_train, x_val, y_val), label_dict


def build_generator(file_label_pairs):
    x = []
    y = []
    for idx in range(0, len(file_label_pairs)):
        file = file_label_pairs[idx][0]
        label = file_label_pairs[idx][1]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            audio_data, sr = librosa.load(file, sr=SAMPLE_RATE)
        n_splits = len(audio_data) // SAMPLE_LIMIT
        audio_data = audio_data[:n_splits * SAMPLE_LIMIT]
        audio_data = np.split(audio_data, list(range(SAMPLE_LIMIT, len(audio_data), SAMPLE_LIMIT)))
        for data in audio_data:
            x.append(data)
            y.append(label)
            if len(data) >= DEFAULT_BATCH_SIZE or idx == len(file_label_pairs) - 1:
                yield np.array(x), np.array(y)
                x = []
                y = []


def build_data_generators(data_directory, validation_split=DEFAULT_VALIDATION_SPLIT, scale_data=True):
    artist_dirs = [os.path.join(data_directory, dir_) for dir_ in os.listdir(data_directory)]
    label_dict = {os.path.basename(artist_dir): idx for idx, artist_dir in enumerate(artist_dirs)}
    labelled_songs = []
    for idx in trange(len(artist_dirs), desc='Creating snippets from artist song samples'):
        artist = os.path.basename(artist_dirs[idx])
        song_files = [os.path.join(artist_dirs[idx], f) for f in os.listdir(artist_dirs[idx]) if '.mp3' in f]
        for file_ in song_files:
            labelled_songs.append([file_, label_dict[artist]])
    random.shuffle(labelled_songs)
    threshold = int(len(labelled_songs)*(1-validation_split))
    training = song_splits[:threshold]
    validation = song_splits[threshold:]
    return build_generator(training), build_generator(validation)


def get_training_data(scale_data=False, save_loaded_data=False, check_saved_data=False):
    load_data = os.path.exists('audio_snippets.pkl') and os.path.exists('labels.pkl') and check_saved_data
    if load_data:
        with open('audio_snippets.pkl', 'rb') as f:
            data = pickle.load(f)
        with open('labels.pkl', 'rb') as f:
            label_dictionary = pickle.load(f)
    else:
        data, label_dictionary = generate_bulk_training_data(DATA_DIR, scale_data=scale_data)
        if save_loaded_data:
            with open('audio_snippets.pkl', 'wb') as f:
                pickle.dump(data, f)
            with open('labels.pkl', 'wb') as f:
                pickle.dump(label_dictionary, f)
    return data, label_dictionary


if __name__ == '__main__':
    training_data, artist_to_label = get_training_data()
    x_train, y_train, x_val, y_val = training_data
    label_to_artist = {v: k for k, v in artist_to_label.items()}

    classifier = build_classifier(input_len=x_train.shape[1], n_classes=y_train.shape[1])
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=1./3.)
    stop_early = tf.keras.callbacks.EarlyStopping(patience=10)

    history = classifier.fit(x_train, y_train, batch_size=DEFAULT_BATCH_SIZE, epochs=100, 
                             validation_data=(x_val, y_val), callbacks=[reduce_lr, stop_early])
    training_history = pd.DataFrame(history.history)
    training_history.to_csv('training_history.csv', index=False)

    classifier.save('song_artist_classifier.h5')

    # Evaluate on validation data and plot confusion matrix
    y_pred = np.argmax(classifier.predict(x_val), axis=1)
    y_true = np.argmax(y_val, axis=1)
    y_pred = [label_to_artist[y] for y in y_pred]
    y_true = [label_to_artist[y] for y in y_true]
    artist_names = list(artist_to_label.keys())
    c_m = ConfusionMatrixDisplay.from_predictions(y_true,
                                                  y_pred,
                                                  labels=artist_names,
                                                  include_values=False,
                                                  xticks_rotation='vertical')
    plt.tight_layout()

    # Plot losses
    plt.figure('Training Losses')
    plt.plot(training_history['loss'])
    plt.plot(training_history['val_loss'])
    plt.legend(['Training', 'Validation'])
    plt.show()


