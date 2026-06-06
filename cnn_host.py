import numpy as np
import os
import sys
import ctypes
from tqdm import tqdm

loma_public_path = os.path.join(os.path.dirname(__file__), '..', 'loma_public')
sys.path.insert(0, loma_public_path)
import compiler

C_FLOAT_PTR = ctypes.POINTER(ctypes.c_float)

def get_loma_func():
    with open('../loma_code/cnn.py') as f:
        _, lib = compiler.compile(f.read(),
                                  target = 'c',
                                  output_filename = '_code/cnn')

    grad_f = lib.grad_cnn
    forward_loss = lib.cnn_loss

    return grad_f, forward_loss

def train_cnn(noisy_audio, clean_audio, N, K=20, epochs=250, lr=1e-5, BATCH_SIZE=32):
    grad_f, forward_loss = get_loma_func() 
    
    weights = np.zeros(K, dtype=np.float32)
    bias = 1.0

    loss_history = []

    for epoch in tqdm(range(epochs), "training..."):
        num_samples = len(noisy_audio)
        shuffled_indices = np.random.permutation(num_samples)
        
        batched_noisy_data = []
        batched_clean_data = []
        
        for i in range(0, num_samples, BATCH_SIZE):
            batch_indices = shuffled_indices[i:i+BATCH_SIZE]
            batched_noisy_data.append([noisy_audio[j] for j in batch_indices])
            batched_clean_data.append([clean_audio[j] for j in batch_indices])

        batched_data = list(zip(batched_noisy_data, batched_clean_data))
        batch_loss = []

        for noisy_batch, clean_batch in batched_data:
            grad = np.zeros(K, dtype=np.float32)
            grad_bias = ctypes.c_float(0.0)
            
            entry_loss = []
            
            for idx in range(len(noisy_batch)):
                entry_noisy = noisy_batch[idx].astype(np.float32) if hasattr(noisy_batch[idx], 'astype') else np.array([noisy_batch[idx]], dtype=np.float32)
                entry_clean = clean_batch[idx].astype(np.float32) if hasattr(clean_batch[idx], 'astype') else np.array([clean_batch[idx]], dtype=np.float32)

                # dummies
                dummy_d_noisy = np.zeros(len(entry_noisy), dtype=np.float32)
                dummy_d_clean = np.zeros(len(entry_clean), dtype=np.float32)

                grad_f(
                    entry_noisy.ctypes.data_as(C_FLOAT_PTR),
                    entry_clean.ctypes.data_as(C_FLOAT_PTR),
                    weights.ctypes.data_as(C_FLOAT_PTR),
                    bias,
                    grad.ctypes.data_as(C_FLOAT_PTR),
                    grad_bias,
                    dummy_d_noisy.ctypes.data_as(C_FLOAT_PTR),
                    dummy_d_clean.ctypes.data_as(C_FLOAT_PTR),
                    N, K
                )
                current_loss = forward_loss(
                    entry_noisy.ctypes.data_as(C_FLOAT_PTR),
                    entry_clean.ctypes.data_as(C_FLOAT_PTR),
                    weights.ctypes.data_as(C_FLOAT_PTR),
                    bias,
                    N, K
                )

                entry_loss.append(current_loss)
            
            weights -= lr * grad
            bias -= lr * grad_bias.value

            batch_loss.append(np.average(np.array(entry_loss)))

        loss_entry = np.average(np.array(batch_loss))
        loss_history.append(loss_entry)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {loss_entry:.4f}")

    return weights, bias, loss_history


if __name__ == "__main__":
    from data_proc.mock_data import gen_signal
    import os
    import sys
    import numpy as np

    N = 100
    time = np.linspace(0, 10 * np.pi, N, dtype=np.float32)
    data = gen_signal(N, time)

    weights, bias, loss_hist = train_cnn([data[0]], [data[1]], N, epochs=250, BATCH_SIZE=1)

    # print(f"final: weight - {weights}, bias - {bias}")