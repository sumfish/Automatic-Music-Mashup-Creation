import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from scipy.spatial import distance
from scipy import ndimage
from math import sqrt
import madmom.features.beats as bt
import madmom.features.downbeats as dbt

sample = 'musicset/000221.wav'
sample2 = 'musicset/000226.wav'
y, sr = librosa.load(sample)
#y_harmonic, y_percussive = librosa.effects.hpss(y)
tempo, beats = librosa.beat.beat_track(y=y,sr=sr)
print(beats.shape)
onset_env = librosa.onset.onset_strength(y, sr=sr,
                                          aggregate=np.median)
hop_length = 512
plt.figure(figsize=(8, 4))
times = librosa.frames_to_time(np.arange(len(onset_env)),
                                sr=sr, hop_length=hop_length)
plt.plot(times, librosa.util.normalize(onset_env),
          label='Onset strength')
plt.vlines(times[beats], 0, 1, alpha=0.5, color='r',
            linestyle='--', label='Beats')
print(times[beats])

#### madmom
#proc = bt.DBNBeatTrackingProcessor(fps=100)
#act = bt.RNNBeatProcessor()(sample2)
proc = dbt.DBNDownBeatTrackingProcessor(beats_per_bar=4, fps=100)
act = dbt.RNNDownBeatProcessor()(sample)
plt.vlines(proc(act), 0, 1, alpha=0.5, color='g',
            linestyle='--', label='Beats_2')
print(proc(act))

plt.legend(frameon=True, framealpha=0.75)
# Limit the plot to a 15-second window
plt.xlim(0, 15)
plt.gca().xaxis.set_major_formatter(librosa.display.TimeFormatter())
plt.tight_layout()
plt.savefig("beat_vs_onset.png")
plt.show()

