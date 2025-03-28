#!/bin/bash

rm -rf data
rm -rf data-bin
rm -f render.mp4

touch input.txt

echo "$1
data-bin/%d.data
1920 1080 120
7 2 -0.5   1   1     1 1 1   0 0
2 0 0      0.5 0.1   1 1 1   0 0
2 -3 0   0 1 0   1
0 0 1    0 0 1   1
-2 3 0   1 0 0   1
-4 -4 -1 -4 4 -1 4 4 -1 4 -4 -1 1 1 1
10 0 15 0.294118 0.196078 0.0980392 4
" > input.txt

mkdir -p "data-bin"
mkdir -p "data"

make
./main --default < input.txt

python3 conv.py $1
ffmpeg -framerate $1 -i data/%03d.png -pix_fmt yuv420p render.mp4

