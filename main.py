import numpy as np
import matplotlib.pyplot as plt
import pyrubberband as pyrb
import librosa
import os
import mashability as mas

input_phrase='000229.wav'
input_path='./musicset/'
can_path='./smallset/'

def main():
    #input
    print("----input-----")
    input_chroma, input_spect, input_tempo = mas.chroma_and_spectral(input_path+input_phrase)
    stable_rate = mas.harmonic_complex(input_chroma)

    #dataset candidate
    print("----candidate-----")
    V_mashability=0
    pitch_shift=0
    chosed_wave=''
    for candidate in os.listdir(can_path):
        if(candidate.endswith('.wav') and not candidate.endswith(input_phrase)):
            print(candidate+' :')
            S_v,best_pitch=mas.mashibility(input_chroma, input_spect, input_tempo, stable_rate, can_path+candidate)
            
            # choose the best-match
            if(S_v>V_mashability):
                V_mashability=S_v
                pitch_shift=best_pitch
                chosed_wave=candidate   
    print(V_mashability, pitch_shift, chosed_wave)

    #generation
    generation(chosed_wave, pitch_shift, input_chroma)
    
def generation(matched_wave, pitch, input_chroma):
    print("---Generating.....---")
    print("choose:{}".format(matched_wave))
    y, sr = librosa.load(can_path+matched_wave)
    
    # pitch shifting
    y_shift = pyrb.pitch_shift(y, sr, n_steps=-pitch)
    #y_shift = librosa.effects.pitch_shift(y, sr, n_steps=pitch) #by liborsa

    '''
    # checked by hormonic
    y_harmonic, y_percussive = librosa.effects.hpss(y_shift)
    tempo, beat_frames = librosa.beat.beat_track(y=y_percussive,sr=sr)
    chromagram = librosa.feature.chroma_cqt(y=y_harmonic,sr=sr)
    # We'll use the median value of each feature between beat frames                                   
    beat_chroma = librosa.util.sync(chromagram,
                                beat_frames,
                                aggregate=np.median)
    print('shape of can_chroma:{}'.format(beat_chroma.shape))
    can_chroma24 =np.concatenate((beat_chroma,beat_chroma),axis=0)
    mas.harmonic(input_chroma,can_chroma24)
    '''
    
    # time stretch
    #librosa.effects.time_stretch




if __name__ == "__main__":
    main()
    