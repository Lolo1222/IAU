ratio_array=(0.05)
seed_array=(1 2 3 4 5 6 7 8 9 10)

for seed in ${seed_array[*]}
do
    python data.py --ratio=0.05 --seed=${seed}
done