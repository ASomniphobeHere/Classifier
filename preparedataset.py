import librosa
import os
import numpy as np
import time
from pydub import AudioSegment
from pydub.silence import split_on_silence

def remove_silence(audiosample): #returns audio sample with each silence reduced to 100 ms
    audiochunks = split_on_silence(
        audiosample,
        min_silence_len=50,
        silence_thresh=-45,
        keep_silence=10
    )
    result = AudioSegment.empty()
    for chunk in audiochunks:
        result+=chunk
    return result

speakerdata = np.loadtxt('SPEAKERS.txt', comments=';', delimiter='|', usecols=(0, 1), dtype={'names': ('id', 'gender'), 'formats':('i4', 'S1')})#load database
speakerinfo = []
for i in speakerdata[:]:
    speakerinfo.append([i[0], 0 if i[1]==b'M' else 1])#[id, gender] database(male=0, female=1)
speakerinfo = np.asarray(speakerinfo)

counter=0
totlength=0# for metadata

for subdir, dirs, files in os.walk("test-clean"):
    for file in files:

        filename = os.path.join(subdir, file)
        if filename.split('.')[-1] == 'txt':                            #check for txt files
            continue
        speakerid = int(filename.split('\\')[1])                        #get speakerid from filename
        chapterid = '_'.join(filename.split('.')[-2].split('-')[-2:])   #get chapterid from filename
        # print(filename.split('.')[-2], filename.split('.')[-2].split('-')[-2:], chapterid) #for testing filename parts
        listindex = np.where(speakerinfo[:, 0]==speakerid)              #find id in speaker database
        speakergender = speakerinfo[listindex, 1]                       #determine gender from database

        try:
            audio = remove_silence(AudioSegment.from_file(filename))#from flac
            totlength+=audio.duration_seconds#add to total dataset duration
            if speakergender == 1:#female
                exportfilename = 'libriwavtest/f_' + str(speakerid) + '_' + chapterid + '.wav'
            elif speakergender == 0:#male
                exportfilename = 'libriwavtest/m_' + str(speakerid) + '_' + chapterid + '.wav'
            print(speakerid, chapterid)
            audio.export(exportfilename, 'wav') # export as wav to end dir
        except:
            print('Error for speakerid', speakerid, 'and chapterid', chapterid)
            time.sleep(0.5) # for keyboard interrupt
        counter+=1

print(counter, "audio files with total length of ", totlength, "seconds")#metadata