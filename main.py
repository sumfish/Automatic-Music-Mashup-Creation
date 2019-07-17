import matplotlib.pyplot as plt
import mashability as mas
import generation  as gen
import os

input_phrase='000229.wav'
input_path='./musicset/'
can_path='./smallset/'
output_path='./output_audio/'

def main():
    #input
    if not os.path.isdir(output_path+input_phrase[:-4]):
        try:
            os.makedirs(output_path+input_phrase[:-4])
        except FileExistsError:
            print('have existed dir')
                
    print("----input-----")
    input_chroma, input_spect, input_tempo = mas.chroma_and_spectral(input_path+input_phrase)
    print("input tempo:{}".format(input_tempo))
    stable_rate = mas.harmonic_complex(input_chroma)

    '''
    #dataset candidate
    print("----search candidate-----")
    V_mashability=0
    pitch_shift=0
    chosed_wave=''
    for candidate in os.listdir(can_path):
        if(candidate.endswith('.wav') and not candidate.endswith(input_phrase)):
            print('------'+candidate+' :')
            S_v,best_pitch=mas.mashibility(input_chroma, input_spect, input_tempo, stable_rate, can_path+candidate)
            
            # choose the best-match
            if(S_v>V_mashability):
                V_mashability=S_v
                pitch_shift=best_pitch
                chosed_wave=candidate   
    print(V_mashability, pitch_shift, chosed_wave)
    '''
    # generation
    print("---Generating---")
    gen.generation("000225.wav", 10, input_chroma, input_tempo, input_phrase)
    #generation(chosed_wave, pitch_shift, input_chroma, input_tempo)
    

if __name__ == "__main__":
    main()
    