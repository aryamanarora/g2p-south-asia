LANGUAGE := por
MODEL := cond-lstm
POS_CLASS := verb
TAG := V.PTCP\;PST

DATA_DIR := ./data
CHECKPOINT_DIR := ./checkpoints
CHECKPOINT_DIR_LANG := $(CHECKPOINT_DIR)/$(LANGUAGE)
RESULTS_DIR := ./results
RESULTS_DIR_LANG := ./results/$(LANGUAGE)

NORTHEURALEX_DIR := $(DATA_DIR)/northeuralex/
NORTHEURALEX_RAW := $(NORTHEURALEX_DIR)/orig.csv.tar.gz
NORTHEURALEX_EXTRACTED := $(NORTHEURALEX_DIR)/orig.csv
# PROCESSED_DATA_FILE := $(NORTHEURALEX_DIR)/processed.pckl

UNIMORPH_DIR := $(DATA_DIR)/unimorph/$(LANGUAGE)
UNIMORPH_RAW := $(UNIMORPH_DIR)/raw.tsv
PROCESSED_DATA_FILE := $(UNIMORPH_DIR)/processed.pckl

WIKI_DIR := $(DATA_DIR)/wiki/$(LANGUAGE)
WIKI_RAW_FILE := $(WIKI_DIR)/raw.txt
TYPES_DICT_FILE := $(WIKI_DIR)/types.pckl

CHECKPOINT_MODEL_FILE := $(CHECKPOINT_DIR_LANG)/$(MODEL)__nl_2-es_128-hs_128-d_0.3300/model.tch
RESULTS_FILE := $(RESULTS_DIR_LANG)/losses__$(MODEL).csv
WUG_SAMPLES := $(RESULTS_DIR_LANG)/wugs__$(MODEL)__$(POS_CLASS).csv
WUG_SCORES := $(RESULTS_DIR_LANG)/wugs__$(MODEL)__$(TAG).csv

all: process_data get_types train eval sample
	echo "Finished evaluating model"

eval_wugs: $(WUG_SCORES)

sample: $(WUG_SAMPLES)

eval: $(RESULTS_FILE)

train: $(CHECKPOINT_MODEL_FILE)
	echo "Finished training model"

process_data: $(PROCESSED_DATA_FILE)
	echo "Finished processing data"

get_types: $(TYPES_DICT_FILE)

get_unimorph: $(UNIMORPH_RAW)
	echo "Finished processing data"

clean:
	rm -rf $(PROCESSED_DATA_FILE)

$(WUG_SCORES): $(WUG_SAMPLES)
	python src/h03_eval/eval_wugs.py \
	    --dataset sigmorphon17task1 --load smart --arch transformer \
	    --train transducer//task0-data/processed/$(LANGUAGE).trn \
	    --dev transducer//task0-data/processed/$(LANGUAGE).dev \
	    --model transducer/checkpoints/sigmorphon20-task0/transformer/dropout0.3/$(LANGUAGE) \
	    --tgt_file $(WUG_SCORES) --tgt_tag $(TAG) \
	    --wug_file $(WUG_SAMPLES)

$(WUG_SAMPLES): $(CHECKPOINT_MODEL_FILE) $(TYPES_DICT_FILE)
	mkdir -p $(RESULTS_DIR_LANG)
	python src/h03_eval/get_wugs.py \
		--data-path $(DATA_DIR) --checkpoints-path $(CHECKPOINT_DIR) --dict-file $(TYPES_DICT_FILE) \
		--results-path $(RESULTS_DIR) --language $(LANGUAGE) --model $(MODEL) --pos-class $(POS_CLASS)

# Eval language models
$(RESULTS_FILE): $(CHECKPOINT_MODEL_FILE)
	echo "Eval models" $(RESULTS_FILE)
	mkdir -p $(RESULTS_DIR_LANG)
	python src/h03_eval/eval.py --data-path $(DATA_DIR) --checkpoints-path $(CHECKPOINT_DIR) --results-path $(RESULTS_DIR) --language $(LANGUAGE) --model $(MODEL)

# Train types Model
$(CHECKPOINT_MODEL_FILE): $(PROCESSED_DATA_FILE)
	echo "Train lstm models" $(MODEL)
	python src/h02_learn/train.py --data-path $(DATA_DIR) --checkpoints-path $(CHECKPOINT_DIR) --language $(LANGUAGE) --model $(MODEL)
# 	python src/h02_learn/run_languages.py --data-path $(DATA_DIR) --checkpoints-path $(CHECKPOINT_DIR) --model $(MODEL)

# Preprocess Data
$(PROCESSED_DATA_FILE): $(UNIMORPH_RAW)
	echo "Process data"
	python src/h01_data/process_data.py --data-path $(DATA_DIR) --language $(LANGUAGE)

$(TYPES_DICT_FILE):
	mkdir -p $(WIKI_DIR)
	python src/h01_data/get_word_types.py --src-file $(WIKI_RAW_FILE) --dict-file $(TYPES_DICT_FILE)

# Extract data
$(NORTHEURALEX_EXTRACTED):
	echo "Get wiki data"
	tar -C $(NORTHEURALEX_DIR) -xvf $(NORTHEURALEX_RAW)

$(UNIMORPH_RAW):
	mkdir -p $(UNIMORPH_DIR)
	wget https://raw.githubusercontent.com/unimorph/$(LANGUAGE)/master/$(LANGUAGE) -O $(UNIMORPH_RAW)
