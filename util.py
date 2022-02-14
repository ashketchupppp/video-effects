import numpy as np
from functools import partial

def smoothstep(val: int, target: int, steps: int):
  # steps is the number of iterations it will take to get to the target value
  if target > val:
    step = (target - val) / steps
    return val + step
  elif target < val:
    step = (val - target) / steps
    return val - step
  return val

def smoothstep_map(initial_array: np.array, array_to_map_to: np.array, steps: int):
  ''' Smoothly transitions values from intial_array to array_to_map_to, in steps number of steps '''
  assert initial_array.shape == array_to_map_to.shape
  newArray = np.zeros(initial_array.shape)
  for y in range(len(newArray)):
    for x in range(len(newArray[y])):
      for rgb in range(len(newArray[y][x])):
        newArray[y][x][rgb] = smoothstep(initial_array[y][x][rgb], array_to_map_to[y][x][rgb], steps)
  return newArray