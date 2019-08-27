#!/usr/bin/env bash
wget --no-check-certificate -P ./ "http://nlp.stanford.edu/data/glove.6B.zip"
unzip ./glove.6B.zip -d ./glove.6B/
rm ./glove.6B.zip