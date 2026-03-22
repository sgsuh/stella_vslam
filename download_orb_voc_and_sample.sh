#!/bin/bash
curl -sL "https://github.com/stella-cv/FBoW_orb_vocab/raw/main/orb_vocab.fbow" -o build/orb_vocab.fbow

# download a sample dataset from Google Drive
FILE_ID="1d8kADKWBptEqTF7jEVhKatBEdN7g0ikY"
curl -sL "https://drive.usercontent.google.com/download?id=${FILE_ID}&export=download&confirm=t" -o build/aist_living_lab_1.zip
unzip build/aist_living_lab_1.zip -d build/
rm build/aist_living_lab_1.zip