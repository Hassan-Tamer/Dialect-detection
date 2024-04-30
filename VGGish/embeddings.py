from torchvggish import vggish, vggish_input
import numpy as np
import os
from natsort import natsorted

# Initialise model and download weights
embedding_model = vggish()
embedding_model.eval()


X = []
y = []

path = '../Segmented/'
dialects = natsorted(os.listdir(path))
print(f"Number of dialects: {len(dialects)}")
print(dialects)

for i,dialect in enumerate(dialects):
    full_path = path + dialect
    audios = natsorted(os.listdir(full_path))
    for j,audio in enumerate(audios):
        wav_file = full_path + '/' + audio
        try:
            example = vggish_input.wavfile_to_examples(wav_file)
            embeddings = embedding_model.forward(example)
            embeddings = embeddings.detach().numpy()
        except:
            print("ERROR: ", dialect, " AUDIO: ", audio)
            print("CONTINUING...")
            continue

        y.append(i)
        X.append(embeddings)
        print("DIALECT: ", dialect, " AUDIO: ", audio)
    print("FINISHED DIALECT: ", dialect)  

X = np.array(X)
y = np.array(y)

X /=255

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

print("SAVING...")
np.save('X.npy', X)
np.save('y.npy', y)