# Autoencoder Design
# Design Autoencoder for image denoising using Evolutionary Algorithm (Genetic Algorithm).
# Course: Bio-Inspired Computers (BIN)
# Organisation: Brno University of Technology - Faculty of Information Technologies
# Author: Daniel Konecny (xkonec75)
# File: Makefile
# Date: 07. 05. 2021

PY = python3
APP = Main.py
LOGIN = xkonec75
SOURCE_DIR = src
VENV = ../bin-env

all: install run

# Activate venv and install requirements.
install: venv
	$(VENV)/bin/activate && pip install -r requirements.txt

# Create venv if it doesn't exist.
venv:
	test -d $(VENV) || $(PY) -m venv $(VENV)

# Determine if we are in venv and run the app.
run:
	( \
		$(VENV)/bin/activate && pip -V; \
		$(PY) $(SOURCE_DIR)/$(APP); \
	)

clean:
	rm -rf $(VENV)
	find -iname "*.pyc" -delete

pack:
	zip -r $(LOGIN).zip $(SOURCE_DIR)/*.py requirements.txt Makefile *.pdf
