import os
import warnings
import sqlite3
import librosa
import tensorflow as tf
import pandas as pd
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from models import LogSpectrogram
from dataclasses import dataclass
from train_artist_classifier import SAMPLE_RATE, SAMPLE_LIMIT


DB_TABLE_FIELDS = ['Song', 'Artist', 'Embedding']
DEFAULT_SONG_DIRECTORY = './all_artists'


def bytes_to_float32(x):
    return np.frombuffer(x, dtype='float32')


def read_and_split_audio(path):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        audio_data, _ = librosa.load(path, sr=SAMPLE_RATE)
    n_splits = len(audio_data) // SAMPLE_LIMIT
    audio_data = audio_data[:n_splits * SAMPLE_LIMIT]
    audio_data = np.split(audio_data, list(range(SAMPLE_LIMIT, len(audio_data), SAMPLE_LIMIT)))
    return audio_data


@dataclass
class Song:
    song_name: str
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
    def __init__(self, song_dir, db_path='./database/song_database.db'):
        self.encoder = AudioEncoder()
        self.song_dir = song_dir
        self.db_path = db_path
        self.db_connection = sqlite3.connect(db_path)
        self.db_dataframe = None

    def build_database(self):
        self.db_dataframe = pd.DataFrame(columns=DB_TABLE_FIELDS)
        artist_dirs = [os.path.join(self.song_dir, artist) for artist in os.listdir(self.song_dir)]
        for artist_dir in tqdm(artist_dirs, desc='Building database from artist directories', total=len(artist_dirs)):
            song_files = [os.path.join(artist_dir, song) for song in os.listdir(artist_dir)]
            for song_file in song_files:
                audio_slices = read_and_split_audio(song_file)
                embeddings = self.encoder.encode(np.array(audio_slices))
                new_df = pd.DataFrame({'Song':[os.path.basename(song_file)]*len(audio_slices), 
                                       'Artist':[os.path.basename(artist_dir)]*len(audio_slices),
                                       'Embedding':[e.tobytes() for e in np.array(embeddings)]})
                self.db_dataframe = pd.concat([self.db_dataframe, new_df])
        self.db_dataframe.to_sql('song_embeddings', con=self.db_connection, index=False, if_exists='replace')

    def load_database(self, path=None):
        if path is None:
            path = self.db_path
        assert os.path.exists(path), 'Database not found'
        if path != self.db_path:
            self.db_path = path
            self.db_connection = sqlite3.connect(path)
        try:
            self.db_dataframe = pd.read_sql('SELECT * FROM song_embeddings', con=self.db_connection)
            self.db_dataframe['Embedding'] = self.db_dataframe['Embedding'].apply(bytes_to_float32)
        except:
            print('song_embeddings table not found. Has the database been built yet?')

    def find_unique_nearest_neighbours(self, data_point, n_neighbours=5):
        self.db_dataframe['Distance'] = np.linalg.norm(self.db_dataframe['Embedding'].values - data_point)
        results = self.db_dataframe.sort_by(col=['Distance']).head(n_neighbours)
        parsed_results = []
        for idx in range(len(results)):
            result = results.iloc[idx]
            parsed_results.append(Song(result['Song'], result['Artist'], result['Distance']))
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
            if len(unique_results) == 0 or result.song_name != unique_results[-1].song_name:
                unique_results.append(result)
            if len(unique_results) >= n_results:
                break
        return unique_results

    def summary(self):
        if self.db_dataframe is not None:
            n_songs = len(pd.unique(self.db_dataframe['Song']))
            n_artists = len(pd.unique(self.db_dataframe['Artist']))
            print(f'Number of songs: {n_songs}')
            print(f'Number of artists: {n_artists}')
        else:
            print('The somg database has not been loaded yet.')


def build_parser():
    parser = ArgumentParser()

    parser.add_argument('-b', '--build', dest='build', action='store_true',
                        help='Build database table from audio files', required=False)

    parser.add_argument('-r', '--reference', type=str, dest='reference', required=False,
                        default=DEFAULT_SONG_DIRECTORY,
                        help='Directory containing songs to build the database')

    parser.add_argument('-s', '--similarity-search', type=str, dest='similarity_search', 
                        help='Search for similar songs in the database', required=False)

    parser.add_argument('-n', '--n-results', type=int, dest='n_results', default=5,
                        help='Number of results to show during similarity search')

    return parser


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    database = SongDatabase(args.reference)
    if args.build:
        print('\n### Building database\n')
        database.build_database()
        print(database.summary())
    else:
        print('\n### Loading database\n')
        database.load_database()
        print(database.summary())
    if args.similarity_search:
        results = database.get_similar_songs(args.similarity_search, args.n_results)
        print(results)
