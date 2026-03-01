from Formatter import read_memmap, retrieve_metadata 
import matplotlib.pyplot as plt
from lock_in import lock_in
import numpy as np 


filename = "Ejemplo medición/forzado.bin"
sample_frequency, scans, n_channels = retrieve_metadata(filename)

T = 10 
interval = int(T * sample_frequency)
cut_off = 1000 


ts_total = []
phis_total = []
previous = None 
for i in range(0, scans, interval): 
    print(f"Tiempo analizado: {i / interval * T:.2f} s.")
    ts, vs = read_memmap(filename, start=i, end=i+interval)
    amp, phi, new_ts, previous = lock_in(ts, vs, cut_off=cut_off, filter_mode="exponential", trim_output=True, previous=previous, return_previous=True)
    new_ts += 2 * ts_total[-1] - ts_total[0-2] if i else 0
    ts_total = np.append(ts_total, new_ts)
    phis_total = np.append(phis_total, phi)

plt.plot(ts_total, phis_total) 
plt.show() 


