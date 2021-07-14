import csv

trans2ipa = {
    'a': 'ə', '@': 'ə', 'aa': 'ɑː', 'i': 'ɪ', 'ii': 'iː', 'u': 'ʊ', 'uu': 'uː',
    'e': 'eː', 'E': 'ɛː', 'o': 'oː', 'O': 'ɔː',
    '~': '~',

    'k': 'k', 'kh': 'kʰ', 'g': 'g', 'gh': 'gʱ', 'ng': 'ŋ',
    'c': 't͡ʃ', 'ch': 't͡ʃʰ', 'j': 'd͡ʒ', 'jh': 'd͡ʒʱ', 'ny': 'ɲ',
    'tt': 'ʈ', 'tth': 'ʈʰ', 'dd': 'ɖ', 'ddh': 'ɖʱ', 'nn': 'ɳ',
    't': 't̪', 'th': 't̪ʰ', 'd': 'd̪', 'dh': 'd̪ʱ', 'n': 'n',
    'p': 'p', 'ph': 'pʰ', 'b': 'b', 'bh': 'bʱ', 'm': 'm',
    'y': 'j', 'r': 'ɾ', 'l': 'l', 'v': 'ʋ',
    'sh': 'ʃ', 's': 's',
    'h': 'ɦ',

    'q': 'q', 'x': 'x', 'Gh': 'ɣ', 'z': 'z',
    'rr': 'ɽ', 'rrh': 'ɽʱ', 'f': 'f',
    'nq': 'ɴ',

    '-': '-'
}

nasal_assim = {
    'k': 'ng', 'kh': 'ng', 'g': 'ng', 'gh': 'ng', 'ng': 'ng',
    'c': 'ny', 'ch': 'ny', 'j': 'ny', 'jh': 'ny', 'ny': 'ny',
    'tt': 'nn', 'tth': 'nn', 'dd': 'nn', 'ddh': 'nn', 'nn': 'nn',
    't': 'n', 'th': 'n', 'd': 'n', 'dh': 'n', 'n': 'n',
    'p': 'm', 'ph': 'm', 'b': 'm', 'bh': 'm', 'm': 'm',
    'y': '~', 'r': '~', 'l': '~', 'v': '~',
    'sh': 'n', 's': 'n',
    'h': '~',

    'q': 'nq', 'x': 'ng', 'Gh': 'ng', 'z': 'n',
    'rr': '~', 'rrh': '~', 'f': 'm'
}

with open('g2p.csv', 'r') as fin, open('g2p_ipa.csv', 'w') as fout:
    reader = csv.reader(fin)
    for row in reader:
        ipa = row[1].split(' ')
        for i, phone in enumerate(ipa):
            if phone == 'ng':
                if i == len(ipa) - 1: ipa[i] = '~'
                else: ipa[i] = nasal_assim.get(ipa[i + 1], '~')
        ipa = [trans2ipa.get(phone, '?') for phone in ipa]
        row[1] = ' '.join(ipa)
        fout.write(','.join(row) + '\n')
