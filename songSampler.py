import requests
import os
import sys
import json
import re
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import List


@dataclass
class Artist:
    name: str
    genres: List[str]
    endpoint: str


@dataclass
class Track:
    name: str
    endpoint: str
    preview: str


def set_authorization_token():
    if 'SPOTIFY_AUTHORIZATION_TOKEN' not in os.environ:
        os.environ['SPOTIFY_AUTHORIZATION_TOKEN'] = input('Authorization token:')


def download_song_preview(artist_name, url, song_name="sample"):
    response = requests.get(url)
    byte_data = response._content
    if byte_data.__sizeof__ == 0:
        print('Could not fetch audio sample. Please check that the song url is valid.')
    else:
        artist_name = re.sub('[^A-Za-z0-9 ]+', '', artist_name)
        song_name = re.sub('[^A-Za-z0-9 ]+', '', song_name)
        directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'all_artists', artist_name)
        if not os.path.exists(directory):
            os.mkdir(directory)
        file_path = os.path.join(directory, song_name + ".mp3")
        with open(file_path, 'wb') as f:
            f.write(byte_data)
        print(f'{song_name} saved to {file_path}')


def download_artist_song_previews(artist_id):
    ...


class SongSampler:
    def __init__(self, n_songs: int = 20, song_directory=None, artist_ids=None):
        self.token = os.getenv('SPOTIFY_AUTHORIZATION_TOKEN')
        self.n_songs = n_songs
        
        if song_directory:
            self.song_directory = song_directory
        else:
            self.song_directory = os.path.dirname(os.path.realpath(__file__))
        
        self.artist_ids = artist_ids
        
        self.artists = self.get_artists()
        self.top_tracks = self.get_top_tracks(self.artists)
        
    def get_artists(self):
        if self.artist_ids is None:
            url = 'https://api.spotify.com/v1/me/top/artists'
            response = requests.get(
                url,
                headers={
                    "Authorization": f"Bearer {self.token}"
                }
            )
            response = response.text
            response = json.loads(response)
            artists = [
                Artist(item['name'], item['genres'], item['href']) for item in response['items']
            ]
        else:
            assert os.path.exists(self.artist_ids), 'Could not find file with artist ids'
            df = pd.read_csv(self.artist_ids)
            artists = []
            url = 'https://api.spotify.com/v1/artists/'
            for idx in range(len(df)):
                endpoint = url + df.iloc[idx]['id']
                artists.append(Artist(df.iloc[idx]['artist'], df.iloc[idx]['genre'], endpoint))
        return artists
    
    def get_top_tracks(self, artists: list):
        top_tracks = {}
        for artist in artists:
            url = artist.endpoint + '/' + 'top-tracks'
            response = requests.get(
                url,
                headers={
                   "Authorization": f"Bearer {self.token}"
                },
                params={
                    "market": "from_token"
                }
            )
            response = response.text
            response = json.loads(response)
            artist_top_tracks = [Track(item['name'], item['href'], item['preview_url']) for item in response['tracks']]
            top_tracks[artist.name] = artist_top_tracks
        return top_tracks
  
    def get_song_previews(self):
        for _, (artist, tracks) in tqdm(enumerate(self.top_tracks.items()), desc='Downloading song previews for artists'):
            for track in tracks:
                download_song_preview(artist, track.preview, track.name)


class SongLabelDataset:
    def __init__(self, n_songs=10000, use_metadata=True):
        ...

    def build_metadata(self):
        ...

    def save_metadata(self):
        ...

    def download_song_previews(self):
        ...

    def data_generator(self):
        ...


if __name__ == '__main__':
    set_authorization_token()
    sampler = SongSampler(artist_ids='artist_ids.csv')
    sampler.get_song_previews()
