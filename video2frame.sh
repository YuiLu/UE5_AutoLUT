#!/bin/bash


video_path="/data/VCG/videos"

for video in "$video_path/train/"*.mkv; do
    filename=$(basename "$video")
    name="${filename%.*}"

    image_dir="${video/videos/images}"
    image_dir="${image_dir%.*}" 

    mkdir -p "$image_dir"
    ffmpeg -i "$video" -vf "fps=24" "$image_dir/%06d.jpg"
done

for video in "$video_path/test/"*.mkv; do
    filename=$(basename "$video")
    name="${filename%.*}"

    image_dir="${video/videos/images}"
    image_dir="${image_dir%.*}" 

    mkdir -p "$image_dir"
    ffmpeg -i "$video" -vf "fps=24" "$image_dir/%06d.jpg"
done



