import numpy as np
from scipy.io import wavfile
from pydub import AudioSegment
import os
import pickle

SR = 16000 # sample rate 16Hz for everything

def convert_m4a_to_wav(m4a_path, wav_output_path):
    """
    convert voice memo to wav
    """
    print(f"converting {m4a_path} to {wav_output_path}...")
    # load
    audio = AudioSegment.from_file(m4a_path, format="m4a")
    # format
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000)
    
    audio.export(wav_output_path, format="wav")
    print("converted m4a to wav")

def gen_noise(length, sample_rate):
    t = np.arange(length) / sample_rate
    
    # base static
    base_static = np.random.normal(0, 0.025, length)
    
    # random noise every 2 seconds
    swell_envelope = np.sin(2 * np.pi * 0.5 * t) ** 2 
    swell_static = np.random.normal(0, 0.1, length) 
    swelling_noise = swell_static * swell_envelope
    
    # electric noise
    hum_60 = 0.04 * np.sin(2 * np.pi * 60 * t)
    hum_120 = 0.02 * np.sin(2 * np.pi * 120 * t)
    electrical_hum = hum_60 + hum_120
    
    total_noise = base_static + swelling_noise + electrical_hum
    return total_noise.astype(np.float32)

def parse_audio(clean_path, noisy_path=None, write_noise=None, start_n=0, max_n=1000):
    """
    reads wave file. if no noisy path, then add noise to clean. clip at max_n to prevent overflow.
    """
    # load file and convert to mono 16Hz sample
    audio = AudioSegment.from_file(clean_path)
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(SR)
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    clean_audio = samples
    
    # normalize 
    clean_audio = clean_audio.astype(np.float32)
    clean_audio = clean_audio / np.max(np.abs(clean_audio))
    
    if noisy_path:
        _, noisy_audio = wavfile.read(noisy_path)
        noisy_audio = noisy_audio.astype(np.float32)
        noisy_audio = noisy_audio / np.max(np.abs(noisy_audio))
    else:
        # make syn noise
        noise = gen_noise(len(clean_audio), SR)
        noisy_audio = clean_audio + noise
        wavfile.write(write_noise, SR, (noisy_audio * 32767).astype(np.int16))
        
    # return slice to fix in window
    return noisy_audio[start_n:max_n], clean_audio[start_n:max_n]

def save_audio(audio_array, sample_rate, output_path="data/generated.wav"):
    """
    take array of sound data to save
    """
    safe_audio = np.clip(audio_array, -1.0, 1.0)
    audio_int16 = (safe_audio * 32767).astype(np.int16)
    wavfile.write(output_path, sample_rate, audio_int16)
    print(f"saved audio to: {output_path}")


if __name__ == "__main__":
    train_set = []
    test_set = []
    max = 30000

    speeches = ["american", "british", "indian"]
    sizes = [55, 83 , 86]

    for l, type_ in zip(sizes, speeches):
        for i in range(l):
            if not os.path.exists(f"data/speech_data/{type_}_accent_{i}.wav"):
                continue
            clean_audio, noisy_audio = parse_audio(f"data/speech_data/{type_}_accent_{i}.wav", f"data/noise_speech_data/{type_}_accent_{i}.wav", max_n=None)
            # sliding window to save audio in sets
            left = 0
            right = max
            while right < len(clean_audio):
                match type_:
                    case "american":
                        if i < 45:
                            train_set.append((clean_audio[left:right], noisy_audio[left:right]))
                        else:
                            test_set.append((clean_audio[left:right], noisy_audio[left:right]))
                    case "british":
                        if i < 68:
                            train_set.append((clean_audio[left:right], noisy_audio[left:right]))
                        else:
                            test_set.append((clean_audio[left:right], noisy_audio[left:right]))
                    case "indian":
                        if i < 71:
                            train_set.append((clean_audio[left:right], noisy_audio[left:right]))
                        else:
                            test_set.append((clean_audio[left:right], noisy_audio[left:right]))
                    case _:
                        print("error, not a case")

                left = right
                right = right + max

    train_set = np.array(train_set)
    test_set = np.array(test_set)
    print("after initial data:")
    print(f"train set size: {train_set.shape}, test set: {test_set.shape}")

    for i in range(1, 5):
        clean_audio, noisy_audio = parse_audio(f"data/clean{i}.wav", f"data/noise{i}.wav", max_n=None)
        left = 0
        right = max
        while right < len(clean_audio):
            add_el = (clean_audio[left:right], noisy_audio[left:right])
            test_set = np.append(test_set, [add_el], axis=0)
            left = right
            right = right + max

    print("after second round data:")
    print(f"train set size: {train_set.shape}, test set: {test_set.shape}")

    with open("data/train_set.pickle", "wb") as f:
        pickle.dump(train_set, f)
    with open("data/test_set.pickle", "wb") as f:
        pickle.dump(test_set, f)
    print("saved train_set and test_set to pickle files")

