#!/bin/bash

GPU=$1
FOLD=${2:-0}

CNTR=0

for F in data/*; do
    CNTR=$((CNTR+1))
    PREFIX=`echo $F | cut -d_ -f1 | sed s_data/__g`
    #if (( CNTR % 1 == FOLD )); then
    #    CUDA_VISIBLE_DEVICES=$GPU python finetune_eval.py $F --epoch 3 --num_exp 1 --model facebook/esm2_t6_8M_UR50D --save /srv/data/berend/szbk/models --prediction_path /srv/data/berend/szbk/ 1>> ft_${FOLD}_${GPU}.log 2> ft_${FOLD}_${GPU}.tmp ;
    #fi
    
    #if [[ $PREFIX == "Phapa1" ]]; then
    CUDA_VISIBLE_DEVICES=$GPU python sparsify.py $F --batch 1024 --layer $FOLD --transformer /srv/data/berend/szbk/models/${PREFIX}_3_0_1_0.0_True_facebook_esm2_t6_8M_UR50D/ 1>> sp_${FOLD}.log 2> sp_${FOLD}.tmp ;
    #fi
    #python sparsify.py $F --layer $FOLD --gpu $GPU --transformer ${PREFIX}_3_0_1_0.0_True_facebook_esm2_t6_8M_UR50D/ >> tmp_sp_${FOLD}.log 2>&1 ;
done
