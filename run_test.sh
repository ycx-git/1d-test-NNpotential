for i in $(seq 1 10); do
    echo "Running test $i"
    export TEMP_VAR=$(i*10)
    python test.py
    # python3 -m unittest discover -s tests -p "test_*.py" > result.log
    # if grep -q "OK" result.log; then
    #     echo "Test $i passed"
    # else
    #     echo "Test $i failed"
    #     break
    # fi
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