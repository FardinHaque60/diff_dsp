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
    with open('../loma_code/soft_threshold.py') as f:
        _, lib = compiler.compile(f.read(),
                                  target = 'c',
                                  output_filename = '_code/soft_threshold')

    grad_f = lib.grad_soft_thresh
    forward_loss = lib.soft_thresh_loss

    return grad_f, forward_loss

def train_soft(noisy_audio, clean_audio, N, epochs=250, lr=1e-3, BATCH_SIZE=32):
    grad_f, forward_loss = get_loma_func() 
    
    thresh = 0.0
    scale = 0.0

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
            thresh_grad = ctypes.c_float(0.0)
            scale_grad = ctypes.c_float(0.0)
            
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
                    thresh,
                    scale,
                    thresh_grad,
                    scale_grad,
                    dummy_d_noisy.ctypes.data_as(C_FLOAT_PTR),
                    dummy_d_clean.ctypes.data_as(C_FLOAT_PTR),
                    N
                )
                current_loss = forward_loss(
                    entry_noisy.ctypes.data_as(C_FLOAT_PTR),
                    entry_clean.ctypes.data_as(C_FLOAT_PTR),
                    thresh,
                    scale,
                    N
                )

                entry_loss.append(current_loss)
            
            thresh -= lr * thresh_grad.value
            scale -= lr * scale_grad.value

            batch_loss.append(np.average(np.array(entry_loss)))

        loss_entry = np.average(np.array(batch_loss))
        loss_history.append(loss_entry)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {loss_entry:.4f}")

    print(f"final loss: {loss_history[-1]}")
    return thresh, scale, loss_history


if __name__ == "__main__":
    from data_proc.mock_data import gen_signal
    import os
    import sys
    import numpy as np

    N = 100
    time = np.linspace(0, 10 * np.pi, N, dtype=np.float32)
    data = gen_signal(N, time)

    thresh, scale, loss_hist = train_soft([data[0]], [data[1]], N, epochs=250, BATCH_SIZE=1)

    print(f"final: thresh - {thresh}, scale z- {scale}")