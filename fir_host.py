import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import ctypes

loma_public_path = os.path.join(os.path.dirname(__file__), '..', 'loma_public')
sys.path.insert(0, loma_public_path)
import compiler

# compile loma code
with open('loma_code/fir.py') as f:
    structs, lib = compiler.compile(f.read(),
                                    target = 'c',
                                    output_filename = '_code/fir')

grad_f = lib.grad_fir
forward_loss = lib.fir_loss

N = 1000 
K = 20   

time = np.linspace(0, 10 * np.pi, N, dtype=np.float32)
clean_audio = np.sin(time).astype(np.float32)
noisy_audio = clean_audio + np.random.normal(0, 0.5, N).astype(np.float32)

# init weights
weights = np.random.normal(loc=0.0, scale=0.1, size=K).astype(np.float32)

dummy_d_noisy = np.zeros(N, dtype=np.float32)
dummy_d_clean = np.zeros(N, dtype=np.float32)

C_FLOAT_PTR = ctypes.POINTER(ctypes.c_float)

# TODO try diff learning rates
learning_rate = 0.0000001
epochs = 800
loss_history = []

print("Starting Training...")
for epoch in range(epochs):
    # init grad
    grad_weights = np.zeros(K, dtype=np.float32)
    
    # loma rev pass
    grad_f(
        noisy_audio.ctypes.data_as(C_FLOAT_PTR),
        clean_audio.ctypes.data_as(C_FLOAT_PTR),
        weights.ctypes.data_as(C_FLOAT_PTR),
        grad_weights.ctypes.data_as(C_FLOAT_PTR),
        dummy_d_noisy.ctypes.data_as(C_FLOAT_PTR),
        dummy_d_clean.ctypes.data_as(C_FLOAT_PTR),
        N,
        K
    )
    
    weights -= learning_rate * grad_weights
    current_loss = forward_loss(
        noisy_audio.ctypes.data_as(C_FLOAT_PTR),
        clean_audio.ctypes.data_as(C_FLOAT_PTR),
        weights.ctypes.data_as(C_FLOAT_PTR),
        N, 
        K
    )
    loss_history.append(current_loss)
    
    if epoch % 50 == 0:
        print("weights snippet: ", weights[:10])
        print(f"Epoch {epoch} | Loss: {current_loss:.4f}")

# plot results
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.title("Loss over Time")
plt.plot(loss_history, color='red')

plt.subplot(3, 1, 2)
plt.title("Learned Filter Weights (The FIR Kernel)")
plt.plot(weights, marker='o')

# example
denoised_audio = np.convolve(noisy_audio, weights, mode='same')

plt.subplot(3, 1, 3)
plt.title("Audio Comparison")
plt.plot(noisy_audio[:200], label="Noisy Input", alpha=0.5)
plt.plot(clean_audio[:200], label="Clean Ground Truth", linewidth=2)
plt.plot(denoised_audio[:200], label="Denoised Output", linestyle='--')
plt.legend()

plt.tight_layout()
plt.show()