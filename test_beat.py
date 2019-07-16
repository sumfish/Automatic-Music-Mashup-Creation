import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import madmom.features.beats as bt
import madmom.features.downbeats as dbt
from pydub import AudioSegment #cut

sample = 'musicset/000224.wav'

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
#proc = BeatTrackingProcessor(fps=100)
#proc = bt.DBNBeatTrackingProcessor(fps=100)
#act = bt.RNNBeatProcessor()(sample2)
proc = dbt.DBNDownBeatTrackingProcessor(beats_per_bar=[4, 4], fps=120)
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

# split audio
sound = AudioSegment.from_file(sample)
# cut by miliseconds
opening=0
end=0
has_record=False
count=0
bar4=0
for s, index in proc(act):
    if(index==1):
        if has_record :
            end=s
            bar4=bar4+1
            if(bar4%4==0):
                cut=sound[opening*1000:end*1000]
                cut.export("cut"+str(count)+".wav",format='wav')
                opening=s
                count=count+1
                bar4=0
        else:    
            opening=s
            has_record=True


