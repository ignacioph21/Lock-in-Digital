import numpy as np
import time
import os
from PyIOTech import daq, daqh
from Formatter import get_converted_data
from lock_in_total import get_amplitude_and_phase  

import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
import sys

# === Inicialización de PyQtGraph y layout principal ===
app = QtWidgets.QApplication(sys.argv)

main_widget = QtWidgets.QWidget() 
layout = QtWidgets.QVBoxLayout()
main_widget.setLayout(layout)

# Gráfico principal
win = pg.GraphicsLayoutWidget(title="Fase en tiempo real")
plot = win.addPlot(title="Fase (última muestra por bloque)")
curve = plot.plot(pen='y', symbol='o')
plot.addLine(y=0, pen=pg.mkPen('w', style=QtCore.Qt.DashLine))
plot.setLabel('left', "Fase", units='rad')
plot.setLabel('bottom', "Tiempo", units='s')

# Displays de amplitud y fase
display_layout = QtWidgets.QHBoxLayout()
amp_label = QtWidgets.QLabel("Amplitud: ---")
phase_label = QtWidgets.QLabel("Fase: --- rad")

font = amp_label.font()
font.setPointSize(12)
amp_label.setFont(font)
phase_label.setFont(font)

display_layout.addWidget(amp_label)
display_layout.addWidget(phase_label)

# Agregar a layout principal
layout.addWidget(win)
layout.addLayout(display_layout)

main_widget.show()

# === Parámetros graficador === #
max_dots = 1000
mean_length = 10
subsample = 4000
fc = 10.0500

# === Configuración DAQ ===
device_name = b'PersonalDaq3001{374679}'
dev = daq.daqDevice(device_name)

flags = daqh.DafAnalog | daqh.DafBipolar | daqh.DafDifferential | daqh.DafSettle1us
gain = daqh.DgainX1
freq = int(500000)
buf_size = 8192
n_channels = 2

dev.AdcSetScan([0, 1], [gain] * n_channels, [flags] * n_channels)
dev.AdcSetFreq(freq)
actual_freq = dev.AdcGetFreq()
print(f"Frecuencia real de muestreo por canal: {actual_freq:.1f} Hz")

dev.AdcSetAcq(daqh.DaamInfinitePost, 0, 0)
dev.AdcSetTrig(daqh.DatsSoftware, 0, 0, 0, 0)
dev.AdcTransferSetBuffer(daqh.DatmUpdateBlock | daqh.DatmCycleOn | daqh.DatmIgnoreOverruns, buf_size)

# === Arranque ===
dev.AdcArm()
dev.AdcTransferStart()
dev.AdcSoftTrig()

print("Adquisición comenzada...")

# === Loop de adquisición infinita ===
try:
    phases = []
    timestamps = []
    t_start = time.time()
    previous = [0, 0, 0, 0]

    while True:
        t0 = time.time() - t_start 
        dev.WaitForEvent(daqh.DteAdcData)
        status = dev.AdcTransferGetStat()

        if status['retCount'] < buf_size:
            continue

        binary_data = dev.dataBuf
        times_arr, voltages_arr = get_converted_data(binary_data, actual_freq, n_channels)

        ts_ref, ref = times_arr[0], voltages_arr[0]
        ts_sig, sig = times_arr[1], voltages_arr[1]

        amp, phase, previous = get_amplitude_and_phase(
            ts_ref, ref, ts_sig, sig, cut_off=fc, return_previous=previous, previous=previous, filter_mode="exponential", random_interval=True
        )

        if len(phases) >= max_dots:
            phases = phases[-max_dots:]
            timestamps = timestamps[-max_dots:]

        phases = np.append(phases, phase[::subsample])
        timestamps = np.append(timestamps, t0 + ts_sig[::subsample]) 

        # Actualizar gráfico
        curve.setData(timestamps, phases)

        # Actualizar displays
        if len(amp) > 0 and len(phase) > 0:
            amp_label.setText(f"Amplitud: {amp[-1]:.4f}")
            phase_label.setText(f"Fase: {phase[-1]:.4f} rad")

        QtWidgets.QApplication.processEvents()

except KeyboardInterrupt:
    print("Adquisición interrumpida por el usuario.")

finally:
    dev.AdcDisarm()
    dev.Close()
    print("Dispositivo cerrado.")
