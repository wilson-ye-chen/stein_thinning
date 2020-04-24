"""Small helper functions."""

import numpy as np

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

def mirror_lower(a):
    i_upper = np.triu_indices(a.shape[0], 1)
    a[i_upper] = a.T[i_upper]
