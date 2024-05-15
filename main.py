from utils import utils
from pytube.innertube import _default_clients

_default_clients["ANDROID_MUSIC"] = _default_clients["ANDROID_CREATOR"]

utls = utils()

path = "https://www.youtube.com/watch?v=XJrj8u2zEk0&ab_channel=%D8%A7%D9%84%D8%AD%D9%83%D8%A7%D9%8A%D8%A9"
# utls.download_playlist(path ,2)
# utls.download_video(path,1)
# utls.get_total_duration("/home/hassan/dataset/egyptian/",True)
utls.get_total_dataset_duration("/home/hassan/dataset/",True)