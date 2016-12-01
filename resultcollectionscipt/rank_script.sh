#!/bin/bash


method=( 18 19 )
year=( 2001)
word=( 1 2 )

for y in "${year[@]}"
do
	echo $y
	for w in "${word[@]}"
	do
		for i in "${method[@]}"
		do
			x=$(cat ../Data/$y\_hyperparameters.txt | grep -e ^$i' ROUGE-1 Average_R: ' | sed -n $w~2p | sed -e "s/.*: \([0-9\.]*\) .*[\n|)]/\1/g")
			echo $x
		done
		echo
	done
	echo
done
