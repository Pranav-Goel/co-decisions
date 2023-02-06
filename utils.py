import numpy as np
from numpy.linalg import norm

def cosine_sim(vec1, vec2):
    return (vec1 @ vec2.T) / (norm(vec1)*norm(vec2))