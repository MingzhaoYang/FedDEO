#!/usr/bin/env bash

##2.generate images


for i in {0..345..1}
do
echo $i
python generate_images.py --category $i
done
