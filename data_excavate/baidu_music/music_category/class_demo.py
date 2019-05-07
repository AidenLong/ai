# coding:utf-8
from pydub.audio_segment import AudioSegment
from scipy.io import wavfile
from python_speech_features.base import mfcc

# 第一步，MP3=========wav
song = AudioSegment.from_file('./data/3D音效音乐/天空之城.MP3', format='MP3')
song.export('./data/3D音效音乐/红白玫瑰.wav', format='wav')

rate, data = wavfile.read('./data/3D音效音乐/红白玫瑰.wav')
mf_feat = mfcc(data, rate, numcep=13, nfft=2048)
# 进一步的降维处理，便于后续的机器学习使用，这里会涉及到一些数学经验算法
