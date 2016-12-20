#!/bin/bash


method=( 2 84 1 80 5 82 6 83 86 85 )
year=(2001)
word=( 1 )

for y in "${year[@]}"
do
	for w in "${word[@]}"
	do
		for i in "${method[@]}"
		do
			x=$(cat $1 | grep -e ^$i' ROUGE-1 Average_R: ' | sed -n $w~1p | sed -e "s/.*: \([0-9\.]*\) .*[\n|)]/\1/g")
			echo $x | tr " " ","
		done
		echo
	done
	echo
done
