import os 
from pydub import AudioSegment
from pytube import Playlist
from pytube import YouTube

class utils:

    def get_total_dataset_duration(self, path ,verbose=False):
        """
        Returns the total length of all the wav files in all directories in the given path
        """
        dialects = os.listdir(path)
        dict_dialects = {}
        for dialect in dialects:
            dir_path = path + dialect + "/"
            length = self.get_total_duration(dir_path,False)
            if verbose:
                print(f"{dialect}: {length} minutes")
            dict_dialects[dialect] = length
            

        if verbose:
            # print(dict_dialects)
            pass
        return dict_dialects
            

    def get_total_duration(self, path ,verbose=False):
        """
        Returns the total length of all the wav files in the directory
        """
        list_dir = os.listdir(path)
        length = 0
        for url in list_dir:
            if url.endswith('.wav'):                
                audio = AudioSegment.from_file(path+url, format="wav")
                length += len(audio)
        if verbose:
            print(f"Total length of all wav files in the directory: {length/1000/60} minutes")
        return length/1000/60
    
    def download_video(self,url,trim=3):
        """
        Downloads the video from the given URL and trims the audio by {trim} minutes from the start and end
        trim: int, default=3
        """
        try:
            video = YouTube(url)
            stream = video.streams.filter(only_audio=True).first()
            print(f"Downloading {video.title}")
            file_path = stream.download(filename=f"{video.title}.mp4")
            print(f"{video.title} is downloaded in WAV format")
            try:
                audio = AudioSegment.from_file(file_path, format="mp4")
            except FileNotFoundError:
                print("Error: File not found")
                return
            if len(audio) < trim * 2  * 60 * 1000:
                print(f"The audio is less than {trim * 2} minutes long.")
                print("Deleting the file.")
                os.remove(file_path)
                return
            
            start_time = trim * 60 * 1000  # Convert trim minutes to milliseconds
            end_time = len(audio) - (trim * 60 * 1000)
            trimmed_audio = audio[start_time:end_time]
            trimmed_file_path = f"{video.title}.wav"
            trimmed_audio.export(trimmed_file_path, format="wav")
            print(f"Trimmed audio saved as {trimmed_file_path}")
        except Exception as e:
            print("Unable to fetch video information. Please check the video URL or your network connection.")
            print(e)
    
    def download_playlist(self,URL_PLAYLIST,trim=3):
        """
        Downloads all the videos in the playlist and trims the audio by {trim} minutes from the start and end
        trim: int, default=3
        """
        playlist = Playlist(URL_PLAYLIST)
        print(f'Number Of Videos In playlist:{len(playlist.video_urls)}')
        urls = []
        for url in playlist:
            urls.append(url)
            
        for i, url in enumerate(urls):
            print(f"Downloading video {i+1} of {len(urls)}")
            print(url)
            self.download_video(url,trim)
            print("\n\n")