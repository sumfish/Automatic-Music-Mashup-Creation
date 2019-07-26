import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from scipy.spatial import distance
from math import sqrt
import math
import faulthandler; faulthandler.enable()
import madmom.audio.chroma as ch
import madmom.features.downbeats as dbt
import madmom.features.tempo as bt
from madmom.features.beats import RNNBeatProcessor 

def mashibility(input_chroma, input_spect, input_tempo, input_down, input_bar, stable_rate, candidate):
    
    use, can_chroma, can_spect, can_tempo, down, bar= chroma_and_spectral(candidate)
    if(use==False): #too short
        return 0, 0
    print('shape of can_chroma:{}'.format(can_chroma.shape))

    #pitch_Shift
    can_chroma24 =np.concatenate((can_chroma,can_chroma),axis=0) #24*beat
    
    '''
    librosa.display.specshow(can_chroma24 ,y_axis='chroma')
    plt.xlabel('beat')
    plt.colorbar()
    plt.title('Chromagram(24*beat)')
    plt.savefig('chroma_24.png')
    '''

    # harmonic similarity
    S_c,pitch = harmonic(input_chroma, can_chroma24, input_down, down)
    
    # harmonic change balance
    W_t = harmonic_balan_w(stable_rate,harmonic_complex(can_chroma))
    #print('change_weight:{}'.format(W_t))

    # spectral 
    
    # tempo
    W_tem=tempo_close_rate(input_tempo,can_tempo)
    
    # final vertical mashability
    S_v=S_c*W_t+W_tem

    return S_v, pitch

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
    stable_rate=float(count)/(i+1)
    #print(len(draw),i+1)
    #print('Stable rate={}'.format(stable_rate))  

    '''
    plt.figure()  
    plt.plot(draw)
    plt.xlabel('beat')
    plt.ylabel('rate')
    plt.title('texture_complex_degree')
    plt.savefig('texture_complex.png')
    '''

    #sigmoid
    complex_degree=1 / (1 + math.exp(-stable_rate))
    #print('after sig rate={}'.format(complex_degree))

    return complex_degree


def harmonic(input_gram, can_gram, input_down, down):
    # chroma index==beat+1?
    #print(input_gram[:,0])
    
    # cosine similiarity
    high_core_pitch=0 
    best_simi=0
    ### ask 13?? +-6+0?
    for pitch in range(12): # pitch-shift 
        cos_simi=0
        for i in range(input_gram.shape[1]-input_down): # axis = num of beat # shape(24,k)
            simi=1-distance.cosine(can_gram[pitch:pitch+12,down+i],input_gram[:,input_down+i]) # compare per beat
            cos_simi+=simi
            ## for input is bigger than candidate(adjust 4 beat per bar)
            if(down+i==can_gram.shape[1]-1):
                break
        cos_simi/=(i+1)

        if(cos_simi>best_simi):
            best_simi=cos_simi
            high_core_pitch=pitch

    #print('best_simi,high_core_pitch:{},{}'.format(best_simi,high_core_pitch))

    return best_simi,high_core_pitch
    
def get_tempo(loop):
    #tempo
    proc2=bt.TempoEstimationProcessor(fps=100)
    act2 = RNNBeatProcessor()(loop)
    tempo=proc2(act2)[0][0]
    return tempo


def chroma_and_spectral(loop):
    ######## madmom
    
    # downbeat
    proc = dbt.DBNDownBeatTrackingProcessor(beats_per_bar= [4,4],fps=100)
    act = dbt.RNNDownBeatProcessor()(loop)
    #print(proc(act)[:,0]) #time v.s. index in a bar
    beat_frames = librosa.time_to_frames(proc(act)[:,0],
                                sr=44100, hop_length=512)   

    # to get downbeat index and segment num 
    # filter music with few bar
    if np.sum(proc(act)[:,1]==1)<3 :
        return False, 0, 0, 0, 0, 0

    downbeat_first=np.argmin(proc(act)[:,1])
    #bug for [1,5,9....] there is a beat before
    if(beat_frames[0]!=0):
        downbeat_first=downbeat_first+1
    bar_num=math.floor((len(beat_frames)-downbeat_first)/4)
    
    '''
    # chroma
    pcp=ch.CLPChromaProcessor(fps=100)
    chroma=pcp(loop)
    chroma=chroma.T #madmom form is different from librosa (frames,12(madmom)) <-> (12,frames(lib))
    '''
    y, sr = librosa.load(loop, sr=44100) # sr=sample rate
    y_harmonic, _ = librosa.effects.hpss(y)
    # beatsynchronous chromagrams.
    chroma = librosa.feature.chroma_cqt(y=y_harmonic,sr=sr,n_chroma=12)
    beat_chroma = librosa.util.sync(chroma,
                                beat_frames,
                                aggregate=np.median)
    #tempo
    tempo=get_tempo(loop)                           
    
    #spec
    beat_spectral=0

    '''
    librosa.display.specshow(beat_chroma,x_axis='chroma')
    plt.title('chromagram(madmom)')
    plt.savefig("chroma(madmom).png")
    '''

    '''
    ######## librosa
    y, sr = librosa.load(loop, sr=44100) # sr=sample rate
    #print(sr)
    
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    tempo, beat_frames = librosa.beat.beat_track(y=y_percussive ,sr=sr)
    print(beat_frames.shape,tempo)
    
    # beatsynchronous chromagrams.
    chromagram = librosa.feature.chroma_cqt(y=y_harmonic,sr=sr,n_chroma=12)
    # We'll use the median value of each feature between beat frames  
             
    beat_chroma = librosa.util.sync(chromagram,
                                beat_frames,
                                aggregate=np.median)

    # beatsynchronous spectralgrams.
    #n_mel?????    
    S=librosa.feature.melspectrogram(y=y,sr=sr,n_mels=7)                         
    beat_spectral = librosa.util.sync(S,
                                beat_frames,
                                aggregate=np.median)
    
    
    librosa.display.specshow(beat_spectral,y_axis='mel')
    plt.xlabel('beat')
    #plt.colorbar(format='%+2.0f dB')
    #plt.colorbar()
    plt.title('Spectrogram')
    plt.savefig(loop+"_spectral_12.png")
    
    librosa.display.specshow(beat_chroma,y_axis='chroma')
    plt.xlabel('beat')
    plt.colorbar()
    plt.title('Chromagram')
    plt.savefig(loop+"_chroma_12.png")
    
    librosa.display.specshow(chromagram,y_axis='chroma')
    plt.xlabel('frame')
    plt.title('Chromagram(no_beat_sync)')
    plt.savefig(loop+"_chroma_nosyc.png")
    '''
    return True, beat_chroma, beat_spectral, tempo, downbeat_first, bar_num
     
'''
if __name__ == "__main__":
    chroma_and_spectral('./musicset/000228.wav')
'''