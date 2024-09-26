#!/bin/bash

# File to be split
input_file="planet_sims/.f6f5e64a-772e-46ab-abac-db75b09ad73c"
shuf $input_file > "${input_file}.shuffled"

# Calculate lines for each set
total_lines=$(wc -l < "${input_file}.shuffled")
train_lines=$(( total_lines * 80 / 100 ))
val_lines=$(( total_lines * 10 / 100 ))
test_lines=$(( total_lines - train_lines - val_lines ))

# Create train, val, and test files
head -n $train_lines "${input_file}.shuffled" > train.txt
tail -n +$(( train_lines + 1 )) "${input_file}.shuffled" | head -n $val_lines > val.txt
tail -n $test_lines "${input_file}.shuffled" > test.txt

# Clean up
rm "${input_file}.shuffled"