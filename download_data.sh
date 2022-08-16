#!/bin/bash
curl -o Input.tar.gz https://weilab.math.msu.edu/Downloads/CLADE2.0/Input.tar.gz
tar -zxvf Input.tar.gz
rsync -a Input/* .
rm -r Input/ 
