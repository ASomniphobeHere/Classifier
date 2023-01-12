import librosa
import os
import numpy as np
import time
from pydub import AudioSegment

speakerdata = np.loadtxt('SPEAKERS.txt', comments=';', delimiter='|', usecols=(0, 1), dtype={'names': ('id', 'gender'), 'formats':('i4', 'S1')})
speakerinfo = []
for i in speakerdata[:]:
    speakerinfo.append([i[0], 0 if i[1]==b'M' else 1])#male 0, female 1
counter=0
totlength=0
speakerinfo = np.asarray(speakerinfo)
for subdir, dirs, files in os.walk("dev-clean"):
    for file in files:
        filename = os.path.join(subdir, file)
        if filename.split('.')[-1] == 'txt':                            #check for txt files
            continue                                                    #if txt skip file
        speakerid = int(filename.split('\\')[1])                        #get speakerid
        chapterid = '_'.join(filename.split('.')[-2].split('-')[-2:])   #get rest of id
        # print(filename.split('.')[-2], filename.split('.')[-2].split('-')[-2:], chapterid) #for testing
        listindex = np.where(speakerinfo[:, 0]==speakerid)              #find id in speaker database
        speakergender = speakerinfo[listindex, 1]                       #determine gender
        try:
            audio = AudioSegment.from_file(filename)
            totlength+=audio.duration_seconds
            if speakergender == 1:#female
                exportfilename = 'libriwavfinal/f_' + str(speakerid) + '_' + chapterid + '.wav'
            elif speakergender == 0:#male
                exportfilename = 'libriwavfinal/m_' + str(speakerid) + '_' + chapterid + '.wav'
            audio.export(exportfilename, 'wav')
        except:
            print('Error for speakerid', speakerid, 'and chapterid', chapterid)
            time.sleep(1) # for keyboard interrupt
        counter+=1
print(counter, "audio files with total length of ", totlength, "seconds")