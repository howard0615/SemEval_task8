model_types=("deberta" "debertav2")
model_names=("microsoft/deberta-large" "microsoft/deberta-v3-large")

data_dir="./data/5-fold"
output_dir="."
train_bz=32
val_bz=32
lr=1e-05
epoch=10

for (( i=0; i< ${#model_types[@]}; i++));
do 
    echo ${model_types[i]};
    echo ${model_names[i]};
    python src/train.py \
        --model_type ${model_types[i]} \
        --model_name ${model_names[i]} \
        --data_dir $data_dir \
        --output_dir $output_dir \
        --train_batch_size $train_bz \
        --val_batch_size $val_bz \
        --grad_acc 1 \
        --lr $lr \
        --epoch $epoch


    python src/val.pt \
        --model_type ${model_types[i]} \
        --model_name ${model_names[i]} \
        --data_dir $data_dir \
        --output_dir $output_dir \
        --train_batch_size $train_bz \
        --val_batch_size $val_bz \
        --grad_acc 1 \
        --lr $lr \
        --epoch $epoch

done


for (( i=0; i< ${#model_types[@]}; i++));
do 
    python src/eval/sen2token_5-fold_evary_chck.py \
        --gold_data_dir $data_dir \
        --model_name ${model_names[i]} \
        --train_batch_size $train_bz \
        --val_batch_size $val_bz \
        --lr $lr \
        --epoch $epoch \
        --grad_acc 1

    python src/eval/eval_script_folder_token_5fold.py \
        --gold_data_dir $data_dir \
        --model_name ${model_names[i]} \
        --train_batch_size $train_bz \
        --val_batch_size $val_bz \
        --lr $lr \
        --epoch $epoch \
        --grad_acc 1

    python src/eval/avg_5fold.py \
        --gold_data_dir $data_dir \
        --model_name ${model_names[i]} \
        --train_batch_size $train_bz \
        --val_batch_size $val_bz \
        --lr $lr \
        --epoch $epoch \
        --grad_acc 1

done
