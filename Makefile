PYTHON ?= python

.PHONY: install download clean corpus labels features train run

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

download:
	$(PYTHON) src/01_download_bref_pbp.py

clean:
	$(PYTHON) src/02_clean_pbp.py

corpus:
	$(PYTHON) src/03_build_corpus.py

labels:
	$(PYTHON) src/04_create_labels.py

features:
	$(PYTHON) src/05_build_features.py

train:
	$(PYTHON) src/06_train_model.py

run:
	$(PYTHON) src/01_download_bref_pbp.py
	$(PYTHON) src/02_clean_pbp.py
	$(PYTHON) src/03_build_corpus.py
	$(PYTHON) src/04_create_labels.py
	$(PYTHON) src/05_build_features.py
	$(PYTHON) src/06_train_model.py
