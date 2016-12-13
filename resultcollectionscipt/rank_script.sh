#!/bin/bash


method=( 82 83 84 )
year=( 2001)
word=( 1 )

for y in "${year[@]}"
do
	echo $y
	for w in "${word[@]}"
	do
		for i in "${method[@]}"
		do
			x=$(cat ../Data/$y\_hyperparameters.txt | grep -e ^$i' ROUGE-1 Average_R: ' | sed -n $w~1p | sed -e "s/.*: \([0-9\.]*\) .*[\n|)]/\1/g")
			echo $x
		done
		echo
	done
	echo
done
