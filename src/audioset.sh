"awkbody='BEGIN{FS=","} {system("youtube-dl -q --no-continue --no-part https://youtube.com/watch?v="$1" -o - 2>/dev/null | ffmpeg -hide_banner -n -i pipe:  -ss "$2" -to "$3" -vf \"scale=256:256:force_original_aspect_ratio=increase,crop=256:256\" -ar 16000 \"./data/test/"$1".mp4\"");}' && cat ./test.csv | grep -v "#.*" | parallel -j32 --bar "echo {} | awk '$awkbody'"
```"
