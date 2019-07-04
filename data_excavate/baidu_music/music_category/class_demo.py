# coding:utf-8
from pydub.audio_segment import AudioSegment
from scipy.io import wavfile
from python_speech_features.base import mfcc

# 第一步，MP3=========wav
song = AudioSegment.from_file('../data/test/你的选择.mp3', format='MP3')
song.export('../data/test/你的选择.wav', format='wav')

rate, data = wavfile.read('../data/test/你的选择.wav')
mf_feat = mfcc(data, rate, numcep=13, nfft=2048)
# 进一步的降维处理，便于后续的机器学习使用，这里会涉及到一些数学经验算法
