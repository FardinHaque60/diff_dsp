def cnn_loss(noisy : In[Array[float]], clean : In[Array[float]], weights : In[Array[float]], bias : In[float], N : In[int], K : In[int]) -> float:
    loss : float = 0.0
    i : int = K - 1
    sum : float = 0.0
    j : int  = 0
    # activated : float = 0.0
    diff : float = 0.0
    
    while (i < N, max_iter := 10000):
        sum = bias
        j = 0
        
        # convolve
        while (j < K, max_iter := 38):
            sum = sum + noisy[i - j] * weights[j]
            j = j + 1
            
        # relu
        # if sum > 0.0:
        #     activated = sum
        # else:
        #     activated = sum * 0.1
        
        diff = sum - clean[i]
        loss = loss + (diff * diff)
        
        i = i + 1
        
    return loss

rev_cnn_loss = rev_diff(cnn_loss)

def grad_cnn(noisy : In[Array[float]], clean : In[Array[float]], 
             weights : In[Array[float]], bias : In[float], 
             grad_weights : Out[Array[float]], grad_bias : Out[float], 
             dummy_d_noisy : Out[Array[float]], dummy_d_clean : Out[Array[float]], 
             N : In[int], K : In[int]):
    
    dret : float = 1.0
    d_N : int = 0
    d_K : int = 0
    
    rev_cnn_loss(
        noisy, dummy_d_noisy, 
        clean, dummy_d_clean, 
        weights, grad_weights, 
        bias, grad_bias,
        N, d_N, 
        K, d_K, 
        dret
    )