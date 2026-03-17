#!/bin/sh

cd `dirname $0`

########
# MINI #
########

mkdir -p puzzles/mini/5x5
mkdir -p puzzles/mini/7x7

for i in $(seq 1 20); do
  outfile="puzzles/mini/5x5/${i}.json"
  if [ ! -f "$outfile" ]; then
    echo "[$i/20] Generating 5x5 mini"
    (uv run crossword-generator generate --type mini --size 5 --seed $i --output-file "$outfile" -v 2>&1) > "puzzles/mini/5x5/${i}-out.txt"
  else
    echo "Skipping (exists): 5x5 mini $i"
  fi
done

for i in $(seq 1 20); do
  outfile="puzzles/mini/7x7/${i}.json"
  if [ ! -f "$outfile" ]; then
    echo "[$i/20] Generating 7x7 mini"
    (uv run crossword-generator generate --type mini --size 7 --seed $i --output-file "$outfile" -v 2>&1) > "puzzles/mini/7x7/${i}-out.txt"
  else
    echo "Skipping (exists): 7x7 mini $i"
  fi
done

########
# MIDI #
########

mkdir -p puzzles/midi/unthemed
mkdir -p puzzles/midi/themed

for i in $(seq 1 20); do
  outfile="puzzles/midi/unthemed/${i}.json"
  if [ ! -f "$outfile" ]; then
    echo "[$i/20] Generating unthemed midi"
    (uv run crossword-generator generate --type midi --no-theme --size 9 --seed $i --output-file "$outfile" -v 2>&1) > "puzzles/midi/unthemed/${i}-out.txt"
  else
    echo "Skipping (exists): unthemed midi $i"
  fi
done

# Generate themes, 200 at a time
# uv run crossword-generator generate-themes --count 200 --size 9 --llm claude

count=0
for theme_file in $(ls themes/*.json | perl -MList::Util=shuffle -e 'print shuffle(<STDIN>)')
do
  [ "$count" -ge 20 ] && break
  base=$(basename "$theme_file" .json)
  outfile="puzzles/midi/themed/${base}.json"
  if [ ! -f "$outfile" ]
  then
    count=$((count + 1))
    echo "[$count/20] Generating themed midi: $base"
    (uv run crossword-generator generate --type midi --size 9 --theme-file "$theme_file" --llm claude --output-file "$outfile" -v 2>&1) > "puzzles/midi/themed/${base}-out.txt"
  else 
    echo "Skipping (exists): $base"
  fi
done


