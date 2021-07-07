# wuggery

[![CircleCI](https://circleci.com/gh/tpimentelms/wuggery.svg?style=svg&circle-token=ee35e2efd77fa72a230df6af16e69c61fd26ced3)](https://circleci.com/gh/tpimentelms/wuggery)

Create new phonotactically plausible wugs


## Install

To install dependencies run:
```bash
$ conda env create -f environment.yml
```

And then install the appropriate version of pytorch:
```bash
$ conda install -y pytorch torchvision cudatoolkit=10.1 -c pytorch
# $ conda install pytorch torchvision cpuonly -c pytorch
```

## Getting tokenized wikipedia data

Use [this Wikipedia tokenizer repository](https://github.com/tpimentelms/wiki-tokenizer) to get the data and move it into `data/wiki/<lang>/parsed.txt` file.


## Generate Wugs

To generate the raw wugs using unimorph data run the command
```bash
$ make LANGUAGE=<lang>
```

## Train Transducer

Go into the folder `transducer/example/sigmorphon2020-shared-tasks/` and follow its README instructions to train the neural-transducer.

## Get Wugs Inflection Entropy

Finally to get the wugs inflection entropy run command
```bash
$ bash src/h03_eval/get_transducer_results.sh <lang>
```
