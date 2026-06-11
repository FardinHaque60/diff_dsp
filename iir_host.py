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
    with open('../loma_code/iir.py') as f:
        _, lib = compiler.compile(f.read(),
                                  target = 'c',
                                  output_filename = '_code/iir')

    grad_f = lib.grad_iir
    forward_loss = lib.iir_loss

    return grad_f, forward_loss

def train_iir(noisy_audio, clean_audio, N, K=31, M=5, epochs=250, lr=1e-5, BATCH_SIZE=32):
    '''
        N is the sample length.
        K is the length of the feedforward weights (b_weights).
        M is the length of the feedback weights (a_weights).
    '''
    grad_f, forward_loss = get_loma_func()

    b_weights = np.zeros(K, dtype=np.float32)
    b_weights[0] = 1.0 
    
    a_weights = np.zeros(M, dtype=np.float32)

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
            grad_b = np.zeros(K, dtype=np.float32)
            grad_a = np.zeros(M, dtype=np.float32)
            
            entry_loss = []
            
            for idx in range(len(noisy_batch)):
                entry_noisy = noisy_batch[idx].astype(np.float32) if hasattr(noisy_batch[idx], 'astype') else np.array([noisy_batch[idx]], dtype=np.float32)
                entry_clean = clean_batch[idx].astype(np.float32) if hasattr(clean_batch[idx], 'astype') else np.array([clean_batch[idx]], dtype=np.float32)

                # dummies
                dummy_d_noisy = np.zeros(len(entry_noisy), dtype=np.float32)
                dummy_d_clean = np.zeros(len(entry_clean), dtype=np.float32)
                dummy_d_out_sig = np.zeros(len(entry_noisy), dtype=np.float32)
                
                # create out sig to save forward pass out
                out_sig_grad = np.zeros(len(entry_noisy), dtype=np.float32)
                out_sig_fwd = np.zeros(len(entry_noisy), dtype=np.float32)

                grad_f(
                    entry_noisy.ctypes.data_as(C_FLOAT_PTR),
                    entry_clean.ctypes.data_as(C_FLOAT_PTR),
                    out_sig_grad.ctypes.data_as(C_FLOAT_PTR),
                    b_weights.ctypes.data_as(C_FLOAT_PTR),
                    a_weights.ctypes.data_as(C_FLOAT_PTR),
                    grad_b.ctypes.data_as(C_FLOAT_PTR),
                    grad_a.ctypes.data_as(C_FLOAT_PTR),
                    dummy_d_noisy.ctypes.data_as(C_FLOAT_PTR),
                    dummy_d_clean.ctypes.data_as(C_FLOAT_PTR),
                    dummy_d_out_sig.ctypes.data_as(C_FLOAT_PTR),
                    N, K, M
                )
                current_loss = forward_loss(
                    entry_noisy.ctypes.data_as(C_FLOAT_PTR),
                    entry_clean.ctypes.data_as(C_FLOAT_PTR),
                    out_sig_fwd.ctypes.data_as(C_FLOAT_PTR),
                    b_weights.ctypes.data_as(C_FLOAT_PTR),
                    a_weights.ctypes.data_as(C_FLOAT_PTR),
                    N, K, M
                )

                entry_loss.append(current_loss)
            
            b_weights -= lr * grad_b
            a_weights -= lr * grad_a # TODO consider making learning rate slower lr * 0.1

            batch_loss.append(np.average(np.array(entry_loss)))

        loss_entry = np.average(np.array(batch_loss))
        loss_history.append(loss_entry)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {loss_entry:.4f}")

    print(f"final loss: {loss_history[-1]}")
    return b_weights, a_weights, loss_history


if __name__ == "__main__":
    from data_proc.mock_data import gen_signal
    from data_proc.parse_audio_data import parse_audio
    import os
    import sys
    import numpy as np

    '''
    N = 1000
    time = np.linspace(0, 10 * np.pi, N, dtype=np.float32)
    data = gen_signal(N, time)
    '''
    N = 1000
    data = parse_audio("data/clean1.wav", "data/noise1.wav")

    fwd_weights, back_weights, loss_hist = train_iir([data[0]], [data[1]], N, K=31, M=5, epochs=250, BATCH_SIZE=1)