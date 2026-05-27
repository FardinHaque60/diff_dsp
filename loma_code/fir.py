def fir_loss(noisy : In[Array[float]], clean : In[Array[float]], weights : In[Array[float]], N : In[int], K : In[int]) -> float:
    loss : float = 0.0
    i : int = K - 1
    sum : float = 0.0
    j : int  = 0
    diff : float = 0
    
    while (i < N, max_iter := 51000): # max value for N
        sum = 0.0
        j = 0
        # convolve
        while (j < K, max_iter := 38): # max value for K
            sum = sum + noisy[i - j] * weights[j]
            j = j + 1
        
        # calc loss
        diff = sum - clean[i]
        loss = loss + (diff * diff)
        i = i + 1
        
    return loss

rev_fir_loss = rev_diff(fir_loss)

def grad_fir(noisy : In[Array[float]], clean : In[Array[float]], weights : In[Array[float]], 
             grad_weights : Out[Array[float]], dummy_d_noisy : Out[Array[float]], dummy_d_clean : Out[Array[float]], 
             N : In[int], K : In[int]):
    
    dret : float = 1.0
    # dummy ints since required in rev pass compilation
    d_N : int = 0
    d_K : int = 0
    
    rev_fir_loss(noisy, dummy_d_noisy, clean, dummy_d_clean, weights, grad_weights, N, d_N, K, d_K, dret)