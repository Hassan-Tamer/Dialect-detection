from torchvggish import vggish, vggish_input
from silence_tensorflow import silence_tensorflow
import numpy as np
import tensorflow as tf

def predict(model,wav_file):
    embedding_model = vggish()
    embedding_model.eval()
    try:
        example = vggish_input.wavfile_to_examples(wav_file)
        embeddings = embedding_model.forward(example)
        embeddings = embeddings.detach().numpy()
    except Exception as e:
        print("ERROR: ", e)
        exit(1)

    embeddings/=255

    y = model.predict(tf.expand_dims(embeddings, axis=0))

    return y

if __name__ == "__main__":
    silence_tensorflow()
    wav_file = '1.wav'
    model = tf.keras.models.load_model('VGGish_model.h5')
    y = predict(model,'1.wav')
    print(y)