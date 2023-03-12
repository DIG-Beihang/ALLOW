#!/bin/sh
module load anaconda/2020.11
module load cuda/10.2
module load cudnn/8.1.0.77_CUDA10.2
module load gcc/5.4
source activate hainanrand
export PYTHONUNBUFFERED=1



DIR=./output/t1_cos_ALF_wo_clu
i=9999
echo "mutual test "$i
    
echo "model_000"$i".pth" > $DIR/last_checkpoint 
    
bash test_t1.sh $DIR

sleep 1m

for((i=19999;i<50000;i+=10000));  
do   
echo "mutual test "$i
    
echo "model_00"$i".pth" > $DIR/last_checkpoint 
    
bash test_t1.sh $DIR

sleep 1m

done

DIR2=./output/t1_half_cycle_ALF
i=999
echo "mutual test "$i
    
echo "model_0000"$i".pth" > $DIR2/last_checkpoint 
    
bash test_t1.sh $DIR2

sleep 1m
for((i=1999;i<10000;i+=1000));  
do   
echo "mutual test "$i
    
echo "model_000"$i".pth" > $DIR2/last_checkpoint 
    
bash test_t1.sh $DIR2

sleep 1m

done

for((i=10999;i<50000;i+=1000));  
do   
echo "mutual test "$i
    
echo "model_00"$i".pth" > $DIR2/last_checkpoint 
    
bash test_t1.sh $DIR2

sleep 1m

done
