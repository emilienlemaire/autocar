'''Examples of organizer-provided metrics.
You can just replace this code by your own.
Make sure to indicate the name of the function that you chose as metric function
in the file metric.txt. E.g. mse_metric, because this file may contain more 
than one function, hence you must specify the name of the function that is your metric.'''

import numpy as np
import scipy as sp

def r2_metric(solution, prediction):
    '''r2_metric error.
    Works even if the target matrix has more than one column'''
    mse = np.mean((prediction - solution)**2)
    var = np.mean((solution - np.mean(solution)) ** 2)
    score = 1 - mse / var
    return score