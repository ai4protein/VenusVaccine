# --------------------case study--------------------
# ElnaggarLab/ankh-large
# facebook/esm2_t33_650M_UR50D
# Rostlab/prot_bert
dataset=VirusBinary
pdb_type=ESMFold
seqs=ez_descriptor,foldseek_seq,esm3_structure_seq
seqs_type=full
plm_group=ElnaggarLab
plm_model=ankh-large
pooling_head=attention1d
lr=5e-4
num_labels=2

CUDA_VISIBLE_DEVICES=0 python eval.py \
    --plm_model ${plm_group}/${plm_model} \
    --dataset $dataset \
    --problem_type single_label_classification \
    --num_labels $num_labels \
    --pooling_method $pooling_head \
    --return_attentions \
    --test_file dataset/CaseStudy/CaseStudy.json \
    --test_result_dir result/$plm_model/$dataset/$seqs_type \
    --metrics auc,accuracy,precision,recall,f1,mcc \
    --structure_seqs $seqs \
    --max_batch_token 10000 \
    --ckpt_root result \
    --ckpt_dir $plm_model/$dataset \
    --model_name "$pdb_type"_"$plm_model"_"$pooling_head"_"$lr"_"$seqs_type"_.pt