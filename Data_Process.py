import os, sys
from PIL import Image
import pdb
# open a pipe from a command
import re
import time
from datetime import datetime
# unlegal='[^A-Za-z\ \']'

def pic_video(file_path, time_ss):
    a, b, c = os.popen3("ffmpeg -i " + file_path)
    # pdb.set_trace()
    out = c.read()
    dp = out.index("Duration: ")
    duration = out[dp + 10:dp + out[dp:].index(",")]
    hh, mm, ss = map(float, duration.split(":"))
    # total time ss
    total = (hh * 60 + mm) * 60 + ss
    season=os.path.basename(file_path)
    if time_ss < total:
        t = time_ss

        # t is seconds in the video
        os.system("ffmpeg -i " + file_path + " -y -f mjpeg -ss %s -t 1 %s_frame_%i.jpg" % (t, season, t))
    return True


def read_srt(data_dir):
    # data_dir = data_dir + '/srt/'
    if not os.path.exists(data_dir):
        print ('data_dir is not exist!')
        return None
    # filelist = []
    # for root, dirs, files in os.walk(data_dir):
    #     for name in files:
    #         file_name = os.path.splitext(os.path.join(root, name))
    #         if file_name[1] == '.json':
    #             filelist.append(os.path.join(root, name))

    # files = os.listdir(data_dir)
    # files = [os.path.join(data_dir, f) for f in files]
    # s = 'Friends.{}'.format(season)
    # one_file = [f for f in files if s in f and 'train' in f][0]
    # for file_ in train_file:

    # pdb.set_trace()
    f = open(data_dir,'r')
    lines = f.readlines()
    f.close()
    times,sentences=[],[]
    TIME_FORMAT='%H:%M:%S,%f'
    # pdb.set_trace()
    start=False
    for no,line in enumerate(lines):
        line=line.strip()
        if line.find('http://bbs.btbbt.com')>0:
            start=True
        if len(line)>0 and start:
            if ' --> 'in line:
                time_begin=line[:line.index(' --> ')]
                time_end=line[line.index(' --> ')+5:]
                # pdb.set_trace()
                time_begin=datetime.strptime(time_begin,TIME_FORMAT)
                time_end = datetime.strptime(time_end, TIME_FORMAT)
                relate_time=time_end - time_begin
                if relate_time.seconds>1:
                    time_=time_begin+(time_end-time_begin)/2
                else:
                    time_=time_end
                times.append(time_)

                sentence=lines[no+1].strip()
                sentences.append(sentence)
    if len(times)==len(sentences):
        return times,sentences
    else:
        pdb.set_trace()
        print('process srt error!')
        return None

def map_time(times, sentences,video_file_path,output_path):
    TIME_FORMAT = '%H:%M:%S,%f'
    output=open(output_path,'w')
    for no, time_ in enumerate(times):
        time_orig = datetime.strptime('00:00:00,0', TIME_FORMAT)
        relate_time = time_ - time_orig
        time_s = relate_time.seconds
        output_time_sents = str(time_) + '\t'+str(time_s)+'\t'+ sentences[no]+'\n'
        output.write(output_time_sents)
    output.close()
    # pdb.set_trace()
    for no,time_ in enumerate(times):
        time_orig = datetime.strptime('00:00:00,0', TIME_FORMAT)
        relate_time = time_ - time_orig
        time_s=relate_time.seconds
        # pdb.set_trace()
        pic_video(video_file_path,time_s)


if __name__ == "__main__":
    video_file_path = '/Users/young/Research/workspace_KG/data/video/S01E01.mp4'
    # pic_video(video_file_path, 32)
    # pdb.set_trace()
    srt_file_path='./data/subtitle/Friends.S01E01.The.One.Where.Monica.Gets.A.New.Roommate.DVDRip.AC3.3.2ch.JOG.srt'
    times,sentences=read_srt(srt_file_path)
    map_time(times,sentences,video_file_path,'./data/frame/S01E01.txt')
