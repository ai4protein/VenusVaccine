#!/bin/bash

# 定义参数
datasets=("BacteriaBinary" "VirusBinary" "TumorBinary")
#datasets=("VirusBinary")

pdb_types=("ESMFold")

seqs_array=("ez_descriptor,foldseek_seq" "foldseek_seq" "ez_descriptor" "None")
seqs_type_array=("full" "foldseek_only" "ez_only" "aa_only")
#seqs_array=("aac,foldseek_seq")
#seqs_type_array=("full_aac")

plm_groups=("facebook" "ElnaggarLab" "Rostlab")
plm_models=("esm2_t33_650M_UR50D" "ankh-base" "prot_bert")
#plm_groups=("Rostlab")
#plm_models=("prot_bert")

pooling_heads=("mean")

lrs=("5e-4")

# 遍历参数
for k in ${!plm_models[@]}; do
  plm_group=${plm_groups[$k]}
  plm_model=${plm_models[$k]}
  for dataset in ${datasets[@]}; do
    for pdb_type in ${pdb_types[@]}; do
      for pooling_head in ${pooling_heads[@]}; do
        for lr in ${lrs[@]}; do
          for i in ${!seqs_array[@]}; do
            seqs=${seqs_array[$i]}
            seqs_type=${seqs_type_array[$i]}

            # 创建一个新的 .pbs 文件
            cp eval_example.pbs eval.pbs

            # 使用 sed 命令修改 .pbs 文件
            sed -i "s/DeepVaccine/DeepVaccine-$dataset/g" eval.pbs
            sed -i "s/out.log/out-$dataset-$pdb_type-$plm_model-$pooling_head-$lr-${seqs_type}.eval.log/g" eval.pbs

            sed -i "s/BacteriaBinary/$dataset/g" eval.pbs
            sed -i "s/ESMFold/$pdb_type/g" eval.pbs
            sed -i "s/ez_descriptor,foldseek_seq/$seqs/g" eval.pbs
            sed -i "s/full/${seqs_type}/g" eval.pbs
            sed -i "s/facebook/$plm_group/g" eval.pbs
            sed -i "s/esm2_t33_650M_UR50D/$plm_model/g" eval.pbs
            sed -i "s/mean/$pooling_head/g" eval.pbs
            sed -i "s/5e-4/$lr/g" eval.pbs
            # 使用 qsub 命令提交任务
            qsub eval.pbs
            echo "Submit task: eval -> DeepVaccine-$dataset-$pdb_type-$plm_model-$pooling_head-$lr-${seqs_type}"
          done
        done
      done
    done
  done
done