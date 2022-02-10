from sklearn.neighbors import KernelDensity
import numpy as np
def TransferEntropy(data, r, slide_x):
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

  
  #data set for history and n+1 point
  X_k = np.zeros((m-1, n-1))
  X_kl = np.zeros((m-1, n-1, 2))
  X_xk = np.zeros((m-1, n-1, 2))
  X_xkl = np.zeros((m-1, n-1, 3))
  
  for i in range(1, m):
    for j in range(1, n):
      X_k[i-1][j-1] = data[i][j-1]
      X_kl[i-1][j-1] = np.array([data[i][j-1], data[i-1][j-1]])
      X_xk[i-1][j-1] = np.array([data[i][j], data[i][j-1]])
      X_xkl[i-1][j-1] = np.array([data[i][j], data[i][j-1], data[i-1][j-1]])

  #flatten the first two diension
  X_k = X_k.reshape((m-1)*(n-1), 1)
  X_kl = X_kl.reshape((m-1)*(n-1), 2)
  X_xk = X_xk.reshape((m-1)*(n-1), 2)
  X_xkl = X_xkl.reshape((m-1)*(n-1), 3)

  kde_k = KernelDensity(kernel='gaussian', bandwidth=r).fit(X_k)
  kde_kl = KernelDensity(kernel='gaussian', bandwidth=r).fit(X_kl)
  kde_xk = KernelDensity(kernel='gaussian', bandwidth=r).fit(X_xk)
  kde_xkl = KernelDensity(kernel='gaussian', bandwidth=r).fit(X_xkl)

  x_min = np.amin(data)
  x_max = np.amax(data)
  Transfer_Entropy = 0

  point_sample = np.arange(x_min, x_max, slide_x)
  for a in point_sample:
    for b in point_sample:
      for c in point_sample:
        k = np.array([[b]])
        kl = np.array([[b, c]])
        xk = np.array([[a, b]])
        xkl = np.array([[a, b, c]])
        pk = np.exp(kde_k.score_samples(k))
        pkl = np.exp(kde_kl.score_samples(kl))
        pxk = np.exp(kde_xk.score_samples(xk))
        pxkl = np.exp(kde_xkl.score_samples(xkl))
        t_ji = pxkl[0]*np.log((pxkl/pkl)/(pxk/pk))[0]
        Transfer_Entropy += t_ji
        #print(pk, pkl, pxk, pxkl)
  return Transfer_Entropy