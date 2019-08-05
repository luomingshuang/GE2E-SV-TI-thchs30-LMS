#coding:utf-8
import glob
import os

thchs30_path = '/home/momozyc/Music/audioData/thchs30/data_thchs30/data'
audio_files = glob.glob(os.path.join(thchs30_path, '*.wav'))
#print(audio_files)
n = 0
audio_speakers = []
for wav_file in audio_files:
    wav_list = wav_file.strip().split('/')
    spkID = wav_list[-1].split('_')[0]
    if spkID not in audio_speakers:
        audio_speakers.append(spkID)
    
#print(len(audio_speakers))
audio_wav_dir = []
print(len(audio_speakers))
print(len(audio_wav_dir))

i = 0
for one in audio_speakers:
    #print(one)
    audio_wav_dir.append([])
    for wav in audio_files:
        wav_list = wav.strip().split('/')
        spkID = wav_list[-1].split('_')[0]
        if spkID == one:
            audio_wav_dir[int(i)].append(wav)
            
    i += 1
            
print(audio_wav_dir, len(audio_wav_dir))

