import csv

with open('ara_atb', 'r') as fin, open('lemmas.csv', 'w') as fout:
    reader = csv.reader(fin, delimiter='\t')
    for row in reader:
        tags = row[2].split(';')
        if 'N' in tags and 'INDF' in tags and row[0] == row[1]:
            fout.write('\t'.join(row) + '\n')