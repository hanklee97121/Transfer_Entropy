import numpy as np
def TransferEntropy(data, r):
  """
  Args: 
    data: np.ndarray(), with shape (m, n)
    r: resolution r
  Output:
    Transfer_Entropy: float
  """
  #space_dim
  m = len(data)
  #time_dim
  n = len(data[0])

  #extract data we need and flatten them
  X_i = data[1:, 1:].flatten() #i_n+1
  X_k = data[1:, :-1].flatten() #i_n^k
  X_l = data[:-1, :-1].flatten() #j_n^l
  
  #data set for history and n+1 point
  X_kl = np.column_stack((X_k, X_l))
  X_xk = np.column_stack((X_i, X_k))
  X_xkl = np.column_stack((X_i, X_k, X_l))
  
  #get bin edges
  data_max = np.amax(data)
  data_min = np.amin(data)
  bin_edges = np.arange(data_min, data_max+r, r)
  
  #get probability density using hist1d, hist2d and histdd
  P_k, _ = np.histogram(X_k, bins=bin_edges, density=True)
  P_kl, _ = np.histogramdd(X_kl, bins=(bin_edges, bin_edges), density=True)
  P_xk, _ = np.histogramdd(X_xk, bins=(bin_edges, bin_edges), density=True)
  P_xkl, _ = np.histogramdd(X_xkl, bins=(bin_edges, bin_edges, bin_edges), density=True)

  #normalize probability density to probability mass
  P_k *= r #(nk)
  P_kl *= r #(nk, nl)
  P_xk *= r #(nx, nk)
  P_xkl *= r #(nx, nk, nl)

  #create conditional probability that will be used to calculate transfer entropy
  p_ik = np.divide(P_xk, P_k, out=np.zeros_like(P_xk), where=P_k!=0) #when P_k=0, output=0 shape = (nx, nk)
  p_ikl = np.divide(P_xkl, P_kl, out=np.zeros_like(P_xkl), where=P_kl!=0) #when P_kl=0, output=0 shape = (nx, nk, nl)
  P_ikl = np.transpose(p_ikl, axes = (2, 0, 1)) #move nl to the first dim for later division shape = (nl, nx, nk)
  p_prelog = np.divide(P_ikl, p_ik, out=np.ones_like(P_ikl), where=p_ik!=0) #when divided by 0, set the output to one because log(1)=0
  p_log = np.log(p_prelog, out=np.zeros_like(p_prelog), where=p_prelog!=0)
  T_ij = P_xkl*np.transpose(p_log, axes = (1, 2, 0)) #shape=(nx, nk, nl)
  T_IJ = np.sum(T_ij)
  return T_IJ