#!/bin/bash

method=( 2 3 4 5 6 7 9 10 11 12 21 1)
year=( 2001 2002 )
word=( 1 2 )

for y in "${year[@]}"
do
	echo $y
	for w in "${word[@]}"
	do
		for i in "${method[@]}"
		do
			x=$(cat ./Data/$y\_hyperparameters.txt | grep -e ^$i' ROUGE-1 Average_R: ' | sed -n $w~2p | sed -e "s/.*: \([0-9\.]*\) .*[\n|)]/\1/g")
			echo $x
		done
		echo
	done
	echo
done
