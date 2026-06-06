def iir_loss(noisy : In[Array[float]], clean : In[Array[float]], out_sig : In[Array[float]], 
             b_weights : In[Array[float]], a_weights : In[Array[float]], 
             N : In[int], K : In[int], M : In[int]) -> float:
    
    loss : float = 0.0
    
    i : int = K
    sum_b : float = 0.0
    sum_a : float = 0.0
    j : int  = 0
    m : int  = 0
    diff : float = 0.0
    
    while (i < N, max_iter := 10000): # original is 51000
        sum_b = 0.0
        j = 0
        while (j < K, max_iter := 38): # orig is 38
            sum_b = sum_b + noisy[i - j] * b_weights[j]
            j = j + 1
            
        sum_a = 0.0
        m = 0
        while (m < M, max_iter := 10):
            # out_sig[i - 1 - m] fetches previous outputs
            sum_a = sum_a + out_sig[i - 1 - m] * a_weights[m]
            m = m + 1
            
        out_sig[i] = sum_b + sum_a
        diff = out_sig[i] - clean[i]
        loss = loss + (diff * diff)
        i = i + 1
        
    return loss

rev_iir_loss = rev_diff(iir_loss)

def grad_iir(noisy : In[Array[float]], clean : In[Array[float]], out_sig : In[Array[float]], 
             b_weights : In[Array[float]], a_weights : In[Array[float]], 
             grad_b_weights : Out[Array[float]], grad_a_weights : Out[Array[float]], 
             dummy_d_noisy : Out[Array[float]], dummy_d_clean : Out[Array[float]], dummy_d_out_sig : Out[Array[float]],
             N : In[int], K : In[int], M : In[int]):
    
    dret : float = 1.0
    d_N : int = 0
    d_K : int = 0
    d_M : int = 0
    
    rev_iir_loss(noisy, dummy_d_noisy, 
                 clean, dummy_d_clean, 
                 out_sig, dummy_d_out_sig,
                 b_weights, grad_b_weights, 
                 a_weights, grad_a_weights, 
                 N, d_N, K, d_K, M, d_M, dret)