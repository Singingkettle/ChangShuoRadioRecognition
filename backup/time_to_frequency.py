import numpy as np
import matplotlib.pyplot as plt

# Number of sample points
N = 1000

# Sample spacing
T = 1.0 / 800.0  # f = 800 Hz

# Create a signal
x = np.linspace(0.0, N * T, N)
t0 = np.pi / 6  # non-zero phase of the second sine
y = np.sin(50.0 * 2.0 * np.pi * x) + 0.5j * np.sin(200.0 * 2.0 * np.pi * x + t0)
yf = np.fft.fft(y)  # to normalize use norm='ortho' as an additional argument

# Where is a 200 Hz frequency in the results?
freq = np.fft.fftfreq(x.size, d=T)
index, = np.where(np.isclose(freq, 200, atol=1 / (T * N)))

# Get magnitude and phase
magnitude = np.abs(yf[index[0]])
phase = np.angle(yf[index[0]])
print("Magnitude:", magnitude, ", phase:", phase)

# Plot a spectrum
plt.plot(freq, 2 / N * np.abs(yf), label='amplitude spectrum')  # in a conventional form
plt.plot(freq, np.angle(yf), label='phase spectrum')
plt.legend()
plt.grid()
plt.show()
