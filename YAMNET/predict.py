import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_io as tfio
import numpy as np

@tf.function
def load_wav_16k_mono(filename):
    """ Load a WAV file, convert it to a float tensor, resample to 16 kHz single-channel audio. """
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(
          file_contents,
          desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav

def predict(model,yamnet_model,wav_file):
    x = load_wav_16k_mono(wav_file)
    scores, embeddings, spectrogram = yamnet_model(x)
    y = model.predict(tf.expand_dims(embeddings, axis=0))
    return y

if __name__ == "__main__":
    yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
    yamnet_model = hub.load(yamnet_model_handle)
    model = tf.keras.models.load_model('yamnet_model.h5')
    y = predict(model,yamnet_model,'1.wav')
    print(y)
