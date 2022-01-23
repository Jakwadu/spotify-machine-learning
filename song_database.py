import os
import warnings
import sqlite3
import librosa
import tensorflow as tf
import pandas as pd
from collections import deque
from models import LogSpectrogram
from dataclasses import dataclass
from train_artist_classifier import SAMPLE_RATE, SAMPLE_LIMIT


DB_TABLE_FIELDS = ['Song', 'Artist', 'Embedding']


def bytes_to_float32(x):
    return np.frombuffer(x, dtype='float32')


def read_and_split_audio(path):
    with warnings.catch_warinings():
        warnings.simplefilter('ignore')
        audio_data, _ = librosa.load(path, sr=SAMPLE_RATE)
    n_splits = len(audio_data) // SAMPLE_LIMIT
    audio_data = audio_data[:n_splits * SAMPLE_LIMIT]
    audio_data = np.split(audio_data, list(range(SAMPLE_LIMIT, len(audio_data), SAMPLE_LIMIT)))
    return audio_data


@dataclass
class Song:
    name: str
    artist: str
    distance: float


class AudioEncoder:
    def __init__(self, saved_model='./database/song_artist_classifier.h5', ):
        base_model = tf.keras.models.load_model(saved_model, custom_objects={"LogSpectrogram":LogSpectrogram})
        self.encoder = tf.keras.Model(inputs=base_model.layers[0].input, outputs=base_model.layers[-3].output)
        self.encoder.trainable = False
        print(self.encoder.summary())

    def encode(self, audio):
        return self.encoder(audio)


class SongDatabase:
    def __init__(self, song_dir='./all_artists', db_path='./database/song_database.db'):
        self.encoder = AudioEncoder()
        self.song_dir = song_dir
        self.db_path = db_path
        self.db_connection = sqlite3.connect(db_path)
        self.db_dataframe = self.build_database()

    def build_database(self):
        df = pd.DataFrame(columns=DB_TABLE_FIELDS)
        artists_dirs = [os.path.join(self.song_dir, artist) for artist in os.path.listdir(self.song_dir)]
        for artist_dir in artist_dirs:
            song_files = [os.path.join(artist_dir, song) for song in os.path.listdir(artist_dir)]
            for song_file in song_files:
                audio_slices = read_and_split_audio(song_file)
                embeddings = self.encoder(np.array([audio_slices]))
                new_df = pd.DataFrame({'Song':[os.path.basename(song_file)]*len(audio_slices), 
                                       'Artist':[os.path.basename(artist_dir)]*len(audio_slices),
                                       'Embedding':[e.tobytes() for e in embeddings]})
                df = pd.concat([df, new_df])
        df.to_sql('song_embeddings', con=self.db_connection, index=False, if_exists='replace')
        return df

    def load_database(self, path=self.db_path):
        assert os.path.exists(path), 'Database not found'
        if path != self.db_path:
            self.db_path = path
            self.db_connection = sqlite3.connect(path)
        try:
            df = pd.read_sql('SELECT * FROM song_embeddings', con=self.db_connection)
            df['Embedding'] = df['Embedding'].apply(bytes_to_float32)
            return df
        except:
            print('song_embeddings table not found. Has the database been built yet?')
            exit()    

    def find_unique_nearest_neighbours(self, data_point, n_neighbours=5):
        self.db_dataframe['Distance'] = np.linalg.norm(self.db_dataframe['Embedding'].values - data_point)
        results = self.db_dataframe.sort_by(col=['Distance']).head(n_neighbours)
        parsed_results = []
        for idx in range(len(results)):
            result = results.iloc[idx]
            parsed_results.append(Song(result['Name'], result['Artist'], result['Distance']))
        return parsed_results
        
    
    def get_similar_songs(self, song_path, n_results=5):
        audio_slices = read_and_split_audio(song_path)
        all_results = []
        for audio_slice in audio_slices:
            results = find_unique_nearest_neighbours(audio_slice, n_neighbours=n_results)
            all_results += results
        sorted(all_results, key=lambda x: x.distance)
        unique_results = []
        for result in all_results:
            if len(unique_results) == 0 or result.name != unique_results[-1].name:
                unique_results.append(result)
            if len(unique_results) >= n_results:
                break
        return unique_results

    def summary(self):
        ...

if __name__ == '__main__':
    AudioEncoder()
