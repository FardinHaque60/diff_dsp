import numpy as np
import os
import sys
import ctypes
from tqdm import tqdm

loma_public_path = os.path.join(os.path.dirname(__file__), '..', 'loma_public')
sys.path.insert(0, loma_public_path)
import compiler

C_FLOAT_PTR = ctypes.POINTER(ctypes.c_float)

# compile loma code
def get_loma_func():
    with open('../loma_code/fir.py') as f:
        _, lib = compiler.compile(f.read(),
                                        target = 'c',
                                        output_filename = '_code/fir')

    grad_f = lib.grad_fir
    forward_loss = lib.fir_loss

    return grad_f, forward_loss

def train_fir(noisy_audio, clean_audio, N, K, epochs=250, lr=0.0000001, BATCH_SIZE=32): # N is how long the longest sample is, K is length of weights vector
    grad_f, forward_loss = get_loma_func()

    # init weights
    weights = np.zeros(K, dtype=np.float32)

    loss_history = []

    for epoch in tqdm(range(epochs), "training..."):
        # partition the data into minibatches of size BATCH_SIZE
        batched_noisy_data = []
        batched_clean_data = []

        num_samples = len(noisy_audio)
        shuffled_indices = np.random.permutation(num_samples)
        for i in range(0, num_samples, BATCH_SIZE):
            batch_indices = shuffled_indices[i:i+BATCH_SIZE]
            batched_noisy_data.append([noisy_audio[j] for j in batch_indices])
            batched_clean_data.append([clean_audio[j] for j in batch_indices])

        # paired batches for iteration
        batched_data = list(zip(batched_noisy_data, batched_clean_data))
        batch_loss = []

        for noisy_batch, clean_batch in batched_data:
            # start of every minibatch reset the grad weight accumulation
            grad_weights = np.zeros(K, dtype=np.float32)
            entry_loss = []
            
            for idx in range(len(noisy_batch)):
                entry_noisy = noisy_batch[idx].astype(np.float32) if hasattr(noisy_batch[idx], 'astype') else np.array([noisy_batch[idx]], dtype=np.float32)
                entry_clean = clean_batch[idx].astype(np.float32) if hasattr(clean_batch[idx], 'astype') else np.array([clean_batch[idx]], dtype=np.float32)

                dummy_d_noisy = np.zeros(len(entry_noisy), dtype=np.float32)
                dummy_d_clean = np.zeros(len(entry_clean), dtype=np.float32)

                # compute the rev pass for every entry in the batch
                grad_f(
                    entry_noisy.ctypes.data_as(C_FLOAT_PTR),
                    entry_clean.ctypes.data_as(C_FLOAT_PTR),
                    weights.ctypes.data_as(C_FLOAT_PTR),
                    grad_weights.ctypes.data_as(C_FLOAT_PTR),
                    dummy_d_noisy.ctypes.data_as(C_FLOAT_PTR),
                    dummy_d_clean.ctypes.data_as(C_FLOAT_PTR),
                    N,
                    K
                )

                current_loss = forward_loss(
                    entry_noisy.ctypes.data_as(C_FLOAT_PTR),
                    entry_clean.ctypes.data_as(C_FLOAT_PTR),
                    weights.ctypes.data_as(C_FLOAT_PTR),
                    N, 
                    K
                )

                entry_loss.append(current_loss)
            # update the weights after the minibatch 
            weights -= lr * grad_weights

            batch_loss.append(np.average(np.array(entry_loss)))

        loss_entry = np.average(np.array(batch_loss))
        loss_history.append(loss_entry)
        
        if epoch % 50 == 0:
            # print("weights snippet: ", weights[:10])
            print(f"Epoch {epoch} | Loss: {loss_entry:.4f}")

    return weights, loss_history