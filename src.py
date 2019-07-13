import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from scipy.spatial import distance
from scipy import ndimage
from math import sqrt
import math

sample = '000221.wav'
sample2 = '000226.wav'
dir_path='musicset/'

def mashibility():
    input_gram,input_tempo = chroma(dir_path+sample2)
    print(input_gram.shape)
    
    can_gram, can_tempo = chroma(dir_path+sample)
    print(can_gram.shape)
    
    #pitch_Shift
    can_gram24 =np.concatenate((can_gram,can_gram),axis=0) #24*beat
    
    '''
    librosa.display.specshow(can_gram24 ,y_axis='chroma')
    plt.xlabel('beat')
    plt.colorbar()
    plt.title('Chromagram(24*beat)')
    plt.savefig('chroma_24.png')
    '''
    # harmonic similarity
    #S_c,pitch = harmonic(input_gram,can_gram)

    # harmonic change balance
    W_t = harmonic_balan_w(harmonic_complex(input_gram),harmonic_complex(can_gram))
    print('change_weight:{}'.format(W_t))

    # spectral 

    # volume
   
    # tempo
    W_tem=tempo_close_rate(input_tempo,can_tempo)

    # final vertical mashability
    #S_v=S_c*W_t+W_tem

def tempo_close_rate(t1,t2):
    if(abs(1-abs(float(t1)/t2))<0.3):
        return 0.2
    else:
        return 0

def harmonic_balan_w(p,q):
    return 1-abs(p-(1-q))

def harmonic_complex(gram):
    #input = 12*beat chromagram
    thre=0.8487 #paper
    count=0 
    #draw=[]
    for i in range(gram.shape[1]-1):
        beat_simi=1-distance.cosine(gram[:,i],gram[:,i+1])
        #draw.append(beat_simi)
        if(beat_simi>thre):
            count=count+1
    rate=float(count)/(i+1)
    #print(len(draw),i+1)
    print('Complex rate={}'.format(rate))  

    '''
    plt.figure()  
    plt.plot(draw)
    plt.xlabel('beat')
    plt.ylabel('rate')
    plt.title('texture_complex_degree')
    plt.savefig('texture_complex.png')
    '''

    #sigmoid
    change_rate=1 / (1 + math.exp(-rate))
    print('after sig rate={}'.format(change_rate))

    return change_rate




def harmonic(input_gram,can_gram):
    # get chroma
    #print(input_gram[:,0])

    # 2D convolution
    # H_n=ndimage.convolve(gram, gram, mode='constant', cval=0.0)
    
    # cosine similiarity
    cos_simi=0
    high_core_pitch=0 
    best_simi=0
    ### ask 13?? +-6+0?
    for pitch in range(12): # pitch-shift 
        for i in range(input_gram.shape[1]): # axis = num of beat # shape(24,k)
            simi=1-distance.cosine(can_gram[pitch:pitch+12,i],input_gram[:,i]) # compare per beat
            cos_simi+=simi
            ## bug for input is bigger than candidate(adjust 4 beat per bar)
            if(i==can_gram.shape[1]-1):
                break

        cos_simi/=(i+1)

        if(cos_simi>best_simi):
            best_simi=cos_simi
            high_core_pitch=pitch

    print(best_simi,high_core_pitch)

    return best_simi,high_core_pitch
    


def chroma(loop):
    # beatsynchronous chromagrams.
    y, sr = librosa.load(loop) # sr=sample rate
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    tempo, beat_frames = librosa.beat.beat_track(y=y_percussive,sr=sr)
    print(beat_frames.shape,tempo)
    chromagram = librosa.feature.chroma_cqt(y=y_harmonic,sr=sr)

    # We'll use the median value of each feature between beat frames                                   
    beat_chroma = librosa.util.sync(chromagram,
                                beat_frames,
                                aggregate=np.median)
    
    '''
    librosa.display.specshow(beat_chroma,y_axis='chroma')
    plt.xlabel('beat')
    plt.colorbar()
    plt.title('Chromagram')
    plt.savefig(loop+"_chroma_12.png")
    
    librosa.display.specshow(chromagram,y_axis='chroma')
    plt.xlabel('beat')
    plt.title('Chromagram(no_beat_sync)')
    plt.savefig(loop+"_chroma_nosyc.png")
    '''
    
    return beat_chroma,tempo


    

if __name__ == "__main__":
    mashibility()
    