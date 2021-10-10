import csv

with open('deu', 'r') as fin, open('lemmas', 'w') as fout, open('g2p.txt', 'w') as fout2:
    reader = csv.reader(fin, delimiter='\t')
    for row in reader:
        tags = row[2].split(';')
        if 'N' in tags and 'NOM' in tags and 'SG' in tags and row[0] == row[1]:
            fout.write('\t'.join(row) + '\n')
            fout2.write(row[0] + '\n')