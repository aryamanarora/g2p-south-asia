from bs4 import BeautifulSoup
import re

conv = {
    '\\': 'ə',
    'Î': 'ɖ', '¯': 'ɳ',
    'ß': 'ʃ',
    'Æ': 'ɭ',
    '˙': 'ʰ'
}

with open('guj-engdictionary.html', 'r') as fin:
    soup = BeautifulSoup(fin.read(), 'html.parser')

for transcription in soup.find_all(text='/'):
    p = list(transcription.parent.get_text().split('/')[1])
    g = list(transcription.parent.get_text().split('<')[1].split('>')[0])
    p = [conv[x] if x in conv else x for x in p]
    g = [conv[x] if x in conv else x for x in g]

    print(p, g)
    input()
    # res = transcription.get_text()
    # text = transcription.parent.parent.parent.find(text=True, recursive=False)
    # print(res, text)
    # input()