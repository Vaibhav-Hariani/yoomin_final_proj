#!/bin/bash
ffmpeg -i $1 -vf scale=96:32 $2