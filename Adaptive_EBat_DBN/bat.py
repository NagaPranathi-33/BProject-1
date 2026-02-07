#!/usr/bin/env python2
import random

# import sys
# import os

# # Add the project root directory to sys.path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# # Now import 

'''from Adaptive_EBat_DBN.BatAlgorithm import *

def Fun(D, sol):
    val = 0.0
    for i in range(D):
        val = val + sol[i] * sol[i]
    return val

# For reproducive results
#random.seed(5)
def algm():
    Algorithm = BatAlgorithm(10, 40, 1000, 0.5, 0.5, 0.0, 2.0, -10.0, 10.0, Fun)
    best = Algorithm.move_bat()
    return abs(best)
'''

import numpy as np
from Adaptive_EBat_DBN.BatAlgorithm import *

def Fun(D, sol):
    """Objective function for optimization"""
    val = sum(sol[i] ** 2 for i in range(D))
    return val

def algm():
    """Runs Bat Algorithm and returns optimized weight matrix."""
    try:
        Algorithm = BatAlgorithm(10, 40, 1000, 0.5, 0.5, 0.0, 2.0, -10.0, 10.0, Fun)
        best = Algorithm.move_bat()  # Best solution from bat algorithm
        
        # ✅ Ensure best is an array
        if isinstance(best, (int, float)):  
            best = np.array([[best]])  # Convert scalar to 2D array

        elif isinstance(best, list):
            best = np.array(best)  # Convert list to NumPy array

        # ✅ Ensure correct shape (modify as needed)
        if best.ndim == 1:
            best = best.reshape(10, 10)  # Reshape based on expected dimensions

        # ✅ Debugging Output
        print(f"✅ Debug: opt_w shape: {best.shape}")
        print(f"✅ Debug: opt_w values:\n{best}")

        return best

    except Exception as e:
        print(f"❌ Error in bat.algm(): {e}")
        return None

