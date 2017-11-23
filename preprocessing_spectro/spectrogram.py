from scipy import signal
import matplotlib.pyplot as plt
from scipy.io import wavfile

rate, data = wavfile.read("/John/data/preprocessing/audio/fn001001.wav")
print(rate)
print(data.shape)
secs= 10
data = data[0: int(secs * rate)]
window = signal.get_window()
f, t, Sxx = signal.spectrogram(data, rate)
plt.pcolormesh(t, f, Sxx)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()