for i in $(seq 1 9); do
    echo "Running test $i"
    export EN_NUM=$i
    CUDA_VISIBLE_DEVICES=$i nohup python -u test_sy_f6.py >> output_$i.log 2>&1 &
done

# CUDA_VISIBLE_DEVICES=9 nohup python -u ./test_f2/test_8.py >> ./test_f2/output_8.log 2>&1 &

# CUDA_VISIBLE_DEVICES=1 nohup python -u test_sy_f6.py >> output_1.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python -u test_sy_f6.py >> output_2.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python -u test_sy_f6.py >> output_3.log 2>&1 &
# CUDA_VISIBLE_DEVICES=4 nohup python -u test_sy_f6.py >> output_4.log 2>&1 &
# CUDA_VISIBLE_DEVICES=5 nohup python -u test_sy_f6.py >> output_5.log 2>&1 &
# CUDA_VISIBLE_DEVICES=6 nohup python -u test_sy_f6.py >> output_6.log 2>&1 &
# CUDA_VISIBLE_DEVICES=7 nohup python -u test_sy_f6.py >> output_7.log 2>&1 &
# CUDA_VISIBLE_DEVICES=8 nohup python -u test_sy_f6.py >> output_8.log 2>&1 &