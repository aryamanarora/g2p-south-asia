
lang=eng
for tag in 'V\;SG\;3\;PRS' 'V\;NFIN' 'V\;PST' 'V.PTCP\;PRS' 'V.PTCP\;PST'
do
    make eval_wugs LANGUAGE=${lang} TAG=${tag}
done

lang=deu
for tag in 'V\;IMP\;SG\;2' 'V\;SBJV\;SG\;3\;PRS' 'V\;IND\;SG\;1\;PRS' 'V\;SBJV\;PL\;2\;PST' 'V\;IND\;PL\;1\;PST' 'V\;IND\;SG\;3\;PST' 'V\;NFIN' 'V\;SBJV\;SG\;3\;PST' 'V\;SBJV\;SG\;1\;PRS' 'V\;IND\;PL\;1\;PRS' 'V\;SBJV\;SG\;2\;PRS' 'V\;IND\;PL\;2\;PRS' 'V\;SBJV\;PL\;3\;PST' 'V\;IND\;SG\;2\;PRS' 'V\;IND\;PL\;3\;PST' 'V\;IND\;SG\;3\;PRS' 'V\;IND\;SG\;2\;PST' 'V\;SBJV\;PL\;1\;PST' 'V\;SBJV\;SG\;2\;PST' 'V\;IND\;PL\;2\;PST' 'V\;SBJV\;PL\;1\;PRS' 'V\;SBJV\;PL\;3\;PRS' 'V\;SBJV\;SG\;1\;PST' 'V\;SBJV\;PL\;2\;PRS' 'V\;IMP\;PL\;2' 'V\;IND\;SG\;1\;PST' 'V\;IND\;PL\;3\;PRS'
do
    make eval_wugs LANGUAGE=${lang} TAG=${tag}
done

lang=deu
pos_class=noun
for tag in 'N\;ACC\;PL' 'N\;NOM\;SG' 'N\;DAT\;SG' 'N\;DAT\;PL' 'N\;NOM\;PL' 'N\;ACC\;SG' 'N\;GEN\;SG' 'N\;GEN\;PL'
do
    make eval_wugs LANGUAGE=${lang} TAG=${tag} POS_CLASS=${pos_class}
done
