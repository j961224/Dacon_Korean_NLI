import re
from itertools import chain
from collections import Counter
from typing import Dict, List, Tuple



def Preprocessor(sentence):
    if "다음 교육에는 더 많은 인원을 대상으로 교육을 실시할 옞ㅇ이다." in sentence:
      sentence = "다음 교육에는 더 많은 인원을 대상으로 교육을 실시할 예정이다."
    if "신 씨 부부는 교통사고를 내서 많은 사상자를 발생시켰다.ㄷ." in sentence:
      sentence = "신 씨 부부는 교통사고를 내서 많은 사상자를 발생시켰다."
    sentence = re.sub("\xa0","",sentence)
    sentence = re.sub("\x08","",sentence)
    sentence = re.sub("《"," ",sentence)
    sentence = re.sub("》"," ",sentence)
    sentence = re.sub("『"," ",sentence)
    sentence = re.sub("』"," ",sentence)
    sentence = re.sub("…"," ",sentence)
    sentence = re.sub('．',".",sentence)
    sentence = ' '.join(sentence.split())
    return sentence