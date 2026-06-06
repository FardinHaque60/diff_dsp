import numpy as np

def gen_signal(N, time):
    # random sine transformation: amplitude, frequency, phase, and vertical shift
    amp = np.random.uniform(0.5, 2.0)
    freq = np.random.uniform(0.5, 2.0)
    phase = np.random.uniform(0, 2 * np.pi)
    shift = np.random.uniform(-1.0, 1.0)

    clean_audio = (amp * np.sin(freq * time + phase) + shift).astype(np.float32)
    noisy_audio = (clean_audio + np.random.normal(0, 0.5, N).astype(np.float32))

    return noisy_audio, clean_audio