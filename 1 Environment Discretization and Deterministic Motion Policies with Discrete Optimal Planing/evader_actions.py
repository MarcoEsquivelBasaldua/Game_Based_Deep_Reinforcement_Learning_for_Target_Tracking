'''
Code name: evader_actions.py
Author: Marco Antonio Esquivel Basaldua

Description:
Predifines the evader actions to be applied in environments Env1 and Env3 (these Env's are used in the DRL improvement)

Input arguments:
    - env: 1 or 3 (deppending on the Env we are interested in)
    - res: not relevant unless definding more behiviours in different resolution configurations

Output:
    - a_e: Dictionary of evader actions given the evader location as key.
'''

import numpy as np

def evader_actions(env, res=1):
    a_e = {}
    if env == 1:
        if res == 1:
            a_e = dict({tuple(np.array([1.5, 1.5])): 3, tuple(np.array([2.5, 1.5])): 3, tuple(np.array([3.5, 1.5])): 0, 
                tuple(np.array([3.5, 2.5])): 0, tuple(np.array([3.5, 3.5])): 0, tuple(np.array([3.5, 4.5])): 0, tuple(np.array([3.5, 5.5])): 0, tuple(np.array([3.5, 6.5])): 0,
                tuple(np.array([3.5, 7.5])): 1, tuple(np.array([2.5, 7.5])): 1, tuple(np.array([1.5, 7.5])): 2,
                tuple(np.array([1.5, 6.5])): 2, tuple(np.array([1.5, 5.5])): 2, tuple(np.array([1.5, 4.5])): 2, tuple(np.array([1.5, 3.5])): 2, tuple(np.array([1.5, 2.5])): 2})

    elif env == 3:
        a_e = dict({tuple(np.array([1.5, 1.5])): 0, tuple(np.array([1.5, 2.5])): 0, tuple(np.array([1.5, 3.5])): 0, tuple(np.array([1.5, 4.5])): 0, tuple(np.array([1.5, 5.5])): 0, tuple(np.array([1.5, 6.5])): 1, 
                tuple(np.array([0.5, 6.5])): 0, tuple(np.array([0.5, 7.5])): 0, tuple(np.array([0.5, 8.5])): 0, tuple(np.array([0.5, 9.5])): 0, tuple(np.array([0.5, 10.5])): 0, tuple(np.array([0.5, 11.5])): 3,
                tuple(np.array([1.5, 11.5])): 2, tuple(np.array([1.5, 10.5])): 2, tuple(np.array([1.5, 9.5])): 3,
                tuple(np.array([2.5, 9.5])): 3, tuple(np.array([3.5, 9.5])): 0, tuple(np.array([3.5, 10.5])): 3, tuple(np.array([4.5, 10.5])): 0, tuple(np.array([4.5, 11.5])): 3,
                tuple(np.array([5.5, 11.5])): 3, tuple(np.array([6.5, 11.5])): 3, tuple(np.array([7.5, 11.5])): 3, tuple(np.array([8.5, 11.5])): 3, tuple(np.array([9.5, 11.5])): 3, tuple(np.array([10.5, 11.5])): 3, tuple(np.array([11.5, 11.5])): 2,
                tuple(np.array([11.5, 10.5])): 2, tuple(np.array([11.5, 9.5])): 3, tuple(np.array([12.5, 9.5])): 2,
                tuple(np.array([12.5, 8.5])): 2, tuple(np.array([12.5, 7.5])): 2, tuple(np.array([12.5, 6.5])): 2, tuple(np.array([12.5, 5.5])): 2, tuple(np.array([12.5, 4.5])): 1,
                tuple(np.array([11.5, 4.5])): 1, tuple(np.array([10.5, 4.5])): 1, tuple(np.array([9.5, 4.5])): 1, tuple(np.array([8.5, 4.5])): 1, tuple(np.array([7.5, 4.5])): 1, tuple(np.array([6.5, 4.5])): 1, tuple(np.array([5.5, 4.5])): 2,
                tuple(np.array([5.5, 3.5])): 2, tuple(np.array([5.5, 2.5])): 2, tuple(np.array([5.5, 1.5])): 2, tuple(np.array([5.5, 0.5])): 1,
                tuple(np.array([4.5, 0.5])): 1, tuple(np.array([3.5, 0.5])): 1, tuple(np.array([2.5, 0.5])): 0, tuple(np.array([2.5, 1.5])): 1 
                })

    return a_e