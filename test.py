import time
import enlighten

pbar = enlighten.Counter(total=100, desc='Basic', unit='ticks')
for num in range(100):
    time.sleep(0.1)  # Simulate work
    pbar.update()