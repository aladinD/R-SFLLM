#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

# NER scripts
python src/run.py -m experiment='glob(ner/*)'

# SC scripts 
python src/run.py -m experiment='glob(ner/*)'

