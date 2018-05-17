import os, sys
from PIL import Image
import pdb
# open a pipe from a command
import re
from datetime import datetime
unlegal='[^A-Za-z\ \']'

def pic_video(file_path, time_ss):
    a, b, c = os.popen3("ffmpeg -i " + file_path)
    pdb.set_trace()
    out = c.read()
    dp = out.index("Duration: ")
    duration = out[dp + 10:dp + out[dp:].index(",")]
    hh, mm, ss = map(float, duration.split(":"))
    # total time ss
    total = (hh * 60 + mm) * 60 + ss
    if time_ss < total:
        t = time_ss

        # t is seconds in the video
        os.system("ffmpeg -i " + file_path + " -y -f mjpeg -ss %s -t 1 %s_frame_%i.jpg" % (t, datetime.now(), t))
    return True


def read_srt(data_dir, season='S01E01'):
    data_dir = data_dir + '/srt/'
    if not os.path.exists(data_dir):
        print ('data_dir is not exist!')
        return None
    # filelist = []
    # for root, dirs, files in os.walk(data_dir):
    #     for name in files:
    #         file_name = os.path.splitext(os.path.join(root, name))
    #         if file_name[1] == '.json':
    #             filelist.append(os.path.join(root, name))

    files = os.listdir(data_dir)
    files = [os.path.join(data_dir, f) for f in files]
    s = 'Friends.{}'.format(season)
    one_file = [f for f in files if s in f and 'train' in f][0]
    # for file_ in train_file:
    f = open(one_file)
    lines = f.readlines()
    f.close()
    times,sentences=[],[]
    for no,line in enumerate(lines):
        if len(line)>0:
            if ' --> 'in line:
                time=line[:line.index(' --> ')]
                times.append(time)
            else:
                sents= re.sub(unlegal, ' ', line)
                if len(sents)>0:
                    sentences.append(line)
    return times,sentences



if __name__ == "__main__":
    file_path = '/Users/young/Downloads/travel0.mov'
    pic_video(file_path, 30)
