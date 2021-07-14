import csv
from collections import Counter

data = {}
with open('../g2p/g2p_ipa.csv', 'r') as fin:
    reader = csv.reader(fin)
    for row in reader:
        data[row[0]] = row[1]

print(f'Found {len(data)} grapheme-phoneme pairs')

endings = []
replacements = []
strange = []

with open('hin_noun.v0.r01.csv', 'r') as fin, open('all_ipa.csv', 'w') as fout:
    reader = csv.reader(fin, delimiter='\t')
    ct, tot = 0, 0
    for row in reader:
        if row[1] in data:
            row[1] = data[row[1]]
            row[1] = row[1].replace(' - ', ' ')
            row[0] = row[1]
            ct += 1
            fout.write('\t'.join(row) + '\n')
        elif row[0] in data:
            if row[1].startswith(row[0]):
                row[1] = row[1][len(row[0]):]
                endings.append(row[1])
            elif row[1].startswith(row[0][:-1]):
                row[1] = row[1][len(row[0]) - 1:]
                replacements.append((row[0][-1], row[1]))
            else:
                strange.append(row)
        tot += 1

print(Counter(endings))
print(Counter(replacements))
print(len(strange))

print(f'Converted {ct} of {tot} lemmas ({ct / tot})')