from silence_tensorflow import silence_tensorflow
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_io as tfio
import pickle

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

if __name__ == '__main__':
    silence_tensorflow()
    yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
    yamnet_model = hub.load(yamnet_model_handle)

    from natsort import natsorted
    import os

    X = []
    y = []

    path = '../Segmented/'
    dialects = natsorted(os.listdir(path))
    print(dialects)

    for i,dialect in enumerate(dialects):
        full_path = path + dialect
        audios = natsorted(os.listdir(full_path))
        for j,audio in enumerate(audios):
            wav_file = full_path + '/' + audio
            try:
                wav_preProcessed = load_wav_16k_mono(wav_file)
            except Exception as e:
                print("ERROR: ", dialect, " AUDIO: ", audio)
                print("CONTINUING...")
                continue
            y.append(i)
            X.append(wav_preProcessed)
            print("DIALECT: ", dialect, " AUDIO: ", audio)
        print("FINISHED DIALECT: ", dialect)  

    print("Extracting Embeddings")
    embeddings_array = []
    for x in X:
        scores, embeddings, spectrogram = yamnet_model(x)
        embeddings_array.append(embeddings)

    file_path = "X.pkl"
    # Save the list to a file using pickle
    with open(file_path, 'wb') as f:
        pickle.dump(embeddings_array, f)

    file_path = 'y.pkl'
    with open(file_path, 'wb') as f:
        pickle.dump(y, f)

    print("DONE")
