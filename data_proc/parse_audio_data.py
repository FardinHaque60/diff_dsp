import numpy as np
from scipy.io import wavfile

def parse_audio(clean_path, noisy_path=None, max_n=1000):
    """
    reads wave file. if no noisy path, then add noise to clean. clip at max_n to prevent overflow.
    """
    sr, clean_audio = wavfile.read(clean_path)
    
    # normalize to [-1.0, 1.0]
    clean_audio = clean_audio.astype(np.float32)
    clean_audio = clean_audio / np.max(np.abs(clean_audio))
    
    if noisy_path:
        _, noisy_audio = wavfile.read(noisy_path)
        noisy_audio = noisy_audio.astype(np.float32)
        noisy_audio = noisy_audio / np.max(np.abs(noisy_audio))
    else:
        # make syn noise
        noise = np.random.normal(0, 0.3, clean_audio.shape).astype(np.float32)
        noisy_audio = clean_audio + noise
        
    # return slice to fix in window
    return noisy_audio[:max_n], clean_audio[:max_n]