from bs4 import BeautifulSoup
import re

conv = {
    '\\': 'a', 'a': 'A',
    'Ô': 'j',
    'ˇ': 'T', 'Î': 'D', '¯': 'N',
    'ß': 'ʃ',
    'Æ': 'ɭ',
    '˙': 'ʰ',
    '.': ''
}

with open('guj-engdictionary.html', 'r') as fin:
    soup = BeautifulSoup(fin.read(), 'html.parser')

def extract(html):
    try:
        p = list(html.parent.get_text().split('/')[1])
        g = list(html.parent.get_text().split('<')[1].split('>')[0])
        p = [conv[x] if x in conv else x for x in p]
        g = [conv[x] if x in conv else x for x in g]
        return "".join(p), "".join(g)
    except:
        return extract(html.parent)

done = set()

with open('guj.tsv', 'w') as fout:
    for transcription in soup.find_all(text='/'):
        p, g = extract(transcription)

        if p not in done:
            fout.write(f'{g}\t{p}\n')
        done.add(p)
        # res = transcription.get_text()
        # text = transcription.parent.parent.parent.find(text=True, recursive=False)
        # print(res, text)
        # input()