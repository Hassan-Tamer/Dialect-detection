from natsort import natsorted
import os
import librosa
import soundfile as sf

# MAKE ALL LENGTHS 1 MINUTE

tot = 0

def segment_audio(name,input_file, output_dir):
    global tot    
    print(tot)
    # Load audio file
    audio, sr = librosa.load(input_file, sr=None)
    
    # Calculate total duration of the audio in seconds
    total_duration = librosa.get_duration(y=audio, sr=sr)
    
    # Define segment duration in seconds (1 minute)
    segment_duration = 60
    
    # Calculate the number of segments
    num_segments = int(total_duration // segment_duration)

    tot += num_segments
    
    # Extract and save each segment
    for i in range(num_segments):
        start_time = i * segment_duration
        end_time = (i + 1) * segment_duration
        segment = audio[int(start_time * sr):int(end_time * sr)]
        output_file = f"{output_dir}/{name}_segment_{i+1}.wav"
        sf.write(output_file, segment, sr)
        print(f"Segment {i+1} saved as {output_file}")
    

X = []
y = []

path = '/home/hassan/dataset/'
dialects = natsorted(os.listdir(path))
print(dialects)

for i,dialect in enumerate(dialects):
    os.makedirs('Segmented/'+dialect, exist_ok=True)
    tot = 0
    full_path = path + dialect
    audios = natsorted(os.listdir(full_path))
    for j,audio in enumerate(audios):
        if tot > 120:
            break
        wav_file = full_path + '/' + audio
        outdir = 'Segmented/'+dialect
        try:
            segment_audio(audio,wav_file, outdir)
        except Exception as e:
            print(e)
            print("ERROR: ", dialect, " AUDIO: ", audio)
            print("CONTINUING...")
            continue

        # y.append(i)
        # X.append(wav_preProcessed)
    print("DIALECT: ", dialect, " AUDIO: ", audio)
    print("FINISHED DIALECT: ", dialect)  