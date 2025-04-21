cd test_sy_f2

cd test_sy_f2_10
nohup python -u test_sy_f2_10.py >> output_10.log 2>&1 &
cd ..

cd test_sy_f2_20
nohup python -u test_sy_f2_20.py >> output_20.log 2>&1 &
cd ..

cd test_sy_f2_30
nohup python -u test_sy_f2_30.py >> output_30.log 2>&1 &
cd ..

cd test_sy_f2_40
nohup python -u test_sy_f2_40.py >> output_40.log 2>&1 &
cd ..

cd test_sy_f2_50
nohup python -u test_sy_f2_50.py >> output_50.log 2>&1 &
cd ..

cd test_sy_f2_60
nohup python -u test_sy_f2_60.py >> output_60.log 2>&1 &
cd ..

cd ..
cd test_sy_f3

cd test_sy_f3_10
nohup python -u test_sy_f3_10.py >> output_10.log 2>&1 &
cd ..
cd test_sy_f3_20
nohup python -u test_sy_f3_20.py >> output_20.log 2>&1 &
cd ..
cd test_sy_f3_30
nohup python -u test_sy_f3_30.py >> output_30.log 2>&1 &
cd ..
cd test_sy_f3_40    
nohup python -u test_sy_f3_40.py >> output_40.log 2>&1 &
cd ..
cd test_sy_f3_50
nohup python -u test_sy_f3_50.py >> output_50.log 2>&1 &
cd ..
cd test_sy_f3_60
nohup python -u test_sy_f3_60.py >> output_60.log 2>&1 &
cd ..

cd ..
cd test_sy_f4

cd test_sy_f4_10
nohup python -u test_sy_f4_10.py >> output_10.log 2>&1 &
cd ..
cd test_sy_f4_20
nohup python -u test_sy_f4_20.py >> output_20.log 2>&1 &
cd ..
cd test_sy_f4_30
nohup python -u test_sy_f4_30.py >> output_30.log 2>&1 &
cd ..
cd test_sy_f4_40
nohup python -u test_sy_f4_40.py >> output_40.log 2>&1 &
cd ..
cd test_sy_f4_50
nohup python -u test_sy_f4_50.py >> output_50.log 2>&1 &
cd ..
cd test_sy_f4_60
nohup python -u test_sy_f4_60.py >> output_60.log 2>&1 &
cd ..

cd ..
cd test_sy_f5

cd test_sy_f5_10
nohup python -u test_sy_f5_10.py >> output_10.log 2>&1 &
cd ..
cd test_sy_f5_20
nohup python -u test_sy_f5_20.py >> output_20.log 2>&1 &
cd ..
cd test_sy_f5_30
nohup python -u test_sy_f5_30.py >> output_30.log 2>&1 &
cd ..
cd test_sy_f5_40    
nohup python -u test_sy_f5_40.py >> output_40.log 2>&1 &
cd ..   
cd test_sy_f5_50
nohup python -u test_sy_f5_50.py >> output_50.log 2>&1 &
cd ..
cd test_sy_f5_60
nohup python -u test_sy_f5_60.py >> output_60.log 2>&1 &
cd ..
cd ..






# nohup python task1.py > task1.log 2>&1 &
# PID=$!  # 获取最后一个后台任务的 PID
# wait $PID  # 等待 task1 完成
# echo "任务 1 完成！"