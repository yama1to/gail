#!/bin/bash
mkdir figs gifs
cd prepare

echo "python3 ppomake.py --task $1"
python3 ppomake.py --task $1

echo "python3 collect_data.py --task $1"
python3 collect_data.py --task $1
cd ..
