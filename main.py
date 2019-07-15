import numpy as np
import matplotlib.pyplot as plt
import mashability as mas
import pyrubberband as pyrb
import os

input_phrase='000229.wav'
dir_path='./musicset/'

def main():
    #input
    print("----input-----")
    input_chroma, input_spect, input_tempo = mas.chroma_and_spectral(dir_path+input_phrase)
    stable_rate = mas.harmonic_complex(input_chroma)

    #dataset candidate
    print("----candidate-----")
    V_mashability=0
    pitch_shift=0
    chosed_wave=''
    for candidate in os.listdir(dir_path):
        if(candidate.endswith('.wav')):
            print(candidate+' :')
            S_v,best_pitch=mas.mashibility(input_chroma, input_spect, input_tempo, stable_rate, dir_path+candidate)
            if(S_v>V_mashability):
                V_mashability=S_v
                pitch_shift=best_pitch
                chosed_wave=candidate   
    print(V_mashability, pitch_shift, chosed_wave)
    
def generation():

if __name__ == "__main__":
    main()
    