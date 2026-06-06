def soft_thresh_loss(noisy : In[Array[float]], clean : In[Array[float]], 
                     threshold : In[float], scale : In[float], 
                     N : In[int]) -> float:
    
    loss : float = 0.0
    i : int = 0
    val : float = 0.0
    abs_val : float = 0.0
    sign : float = 0.0
    out_val : float = 0.0
    diff : float = 0.0
    
    while (i < N, max_iter := 51000):
        val = noisy[i]
        
        if val > 0.0:
            abs_val = val
            sign = 1.0
        else:
            abs_val = -val
            sign = -1.0
            
        if abs_val > threshold:
            out_val = sign * (abs_val - threshold) * scale
        else:
            out_val = 0.0
            
        diff = out_val - clean[i]
        loss = loss + (diff * diff)
        
        i = i + 1
        
    return loss

# Generate the backward pass
rev_soft_thresh_loss = rev_diff(soft_thresh_loss)

def grad_soft_thresh(noisy : In[Array[float]], clean : In[Array[float]], 
                     threshold : In[float], scale : In[float],
                     grad_threshold : Out[float], grad_scale : Out[float],
                     dummy_d_noisy : Out[Array[float]], dummy_d_clean : Out[Array[float]],
                     N : In[int]):
    
    dret : float = 1.0
    d_N : int = 0
    
    rev_soft_thresh_loss(noisy, dummy_d_noisy, 
                         clean, dummy_d_clean, 
                         threshold, grad_threshold, 
                         scale, grad_scale, 
                         N, d_N, dret)