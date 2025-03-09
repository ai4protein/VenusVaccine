import argparse
import numpy as np
import torch
import json
import pandas as pd
import re
from transformers import BertTokenizer, BertModel, T5EncoderModel, AutoTokenizer
from transformers import logging
from torch.utils.data import DataLoader
from src.utils.data_utils import BatchSampler
from src.models.adapter import AdapterModel
from src.data.get_esm3_structure_seq import VQVAE_SPECIAL_TOKENS

# 忽略警告信息
logging.set_verbosity_error()

def process_data_line(data, max_seq_len=None):
    if max_seq_len is not None:
        data["aa_seq"] = data["aa_seq"][:max_seq_len]
        if "foldseek_seq" in args.structure_seqs:
            data["foldseek_seq"] = data["foldseek_seq"][:max_seq_len]
        token_num = min(len(data["aa_seq"]), max_seq_len)
    else:
        token_num = len(data["aa_seq"])
    return data, token_num

def process_dataset_from_json(file, max_seq_len=None):
    dataset, token_nums = [], []
    for l in open(file):
        data = json.loads(l)
        data, token_num = process_data_line(data, max_seq_len)
        dataset.append(data)
        token_nums.append(token_num)
    return dataset, token_nums

def e_descriptor_embedding(aa_input_ids):
    aa_seqs = [tokenizer.convert_ids_to_tokens(aa_input_ids[i]) for i in range(len(aa_input_ids))]
    e1 = {'A': 0.008, 'R': 0.171, 'N': 0.255, 'D': 0.303, 'C': -0.132, 'Q': 0.149, 'E': 0.221, 'G': 0.218,
          'H': 0.023, 'I': -0.353, 'L': -0.267, 'K': 0.243, 'M': -0.239, 'F': -0.329, 'P': 0.173, 'S': 0.199,
          'T': 0.068, 'W': -0.296, 'Y': -0.141, 'V': -0.274}
    e2 = {'A': 0.134, 'R': -0.361, 'N': 0.038, 'D': -0.057, 'C': 0.174, 'Q': -0.184, 'E': -0.28, 'G': 0.562,
          'H': -0.177, 'I': 0.071, 'L': 0.018, 'K': -0.339, 'M': -0.141, 'F': -0.023, 'P': 0.286, 'S': 0.238,
          'T': 0.147, 'W': -0.186, 'Y': -0.057, 'V': 0.136}
    e3 = {'A': -0.475, 'R': 0.107, 'N': 0.117, 'D': -0.014, 'C': 0.07, 'Q': -0.03, 'E': -0.315, 'G': -0.024,
          'H': 0.041, 'I': -0.088, 'L': -0.265, 'K': -0.044, 'M': -0.155, 'F': 0.072, 'P': 0.407, 'S': -0.015,
          'T': -0.015, 'W': 0.389, 'Y': 0.425, 'V': -0.187}
    e4 = {'A': -0.039, 'R': -0.258, 'N': 0.118, 'D': 0.225, 'C': 0.565, 'Q': 0.035, 'E': 0.157, 'G': 0.018,
          'H': 0.28, 'I': -0.195, 'L': -0.274, 'K': -0.325, 'M': 0.321, 'F': -0.002, 'P': -0.215, 'S': -0.068,
          'T': -0.132, 'W': 0.083, 'Y': -0.096, 'V': -0.196}
    e5 = {'A': 0.181, 'R': -0.364, 'N': -0.055, 'D': 0.156, 'C': -0.374, 'Q': -0.112, 'E': 0.303, 'G': 0.106,
          'H': -0.021, 'I': -0.107, 'L': 0.206, 'K': -0.027, 'M': 0.077, 'F': 0.208, 'P': 0.384, 'S': -0.196,
          'T': -0.274, 'W': 0.297, 'Y': -0.091, 'V': -0.299}
    e_embeds = [[[e1.get(aa, 0.0), e2.get(aa, 0.0), e3.get(aa, 0.0), e4.get(aa, 0.0), e5.get(aa, 0.0)] for aa in seq] for seq in aa_seqs]
    e_embeds = torch.tensor(e_embeds).float()
    return e_embeds

def z_descriptor_embedding(aa_input_ids):
    aa_seqs = [tokenizer.convert_ids_to_tokens(aa_input_ids[i]) for i in range(len(aa_input_ids))]
    z1 = {'A': 0.07, 'R': 2.88, 'N': 3.22, 'D': 3.64, 'C': 0.71, 'Q': 2.18, 'E': 3.08, 'G': 2.23, 'H': 2.41,
          'I': -4.44, 'L': -4.19, 'K': 2.84, 'M': -2.49, 'F': -4.92, 'P': -1.22, 'S': 1.96, 'T': 0.92, 'W': -4.75,
          'Y': -1.39, 'V': -2.69}
    z2 = {'A': -1.73, 'R': 2.52, 'N': 1.45, 'D': 1.13, 'C': -0.97, 'Q': 0.53, 'E': 0.39, 'G': -5.36, 'H': 1.74,
          'I': -1.68, 'L': -1.03, 'K': 1.41, 'M': -0.27, 'F': 1.30, 'P': 0.88, 'S': -1.63, 'T': -2.09, 'W': 3.65,
          'Y': 2.32, 'V': -2.53}
    z3 = {'A': 0.09, 'R': -3.44, 'N': 0.84, 'D': 2.36, 'C': 4.13, 'Q': -1.14, 'E': -0.07, 'G': 0.30, 'H': 1.11,
          'I': -1.03, 'L': -0.98, 'K': -3.14, 'M': -0.41, 'F': 0.45, 'P': 2.23, 'S': 0.57, 'T': -1.40, 'W': 0.85,
          'Y': 0.01, 'V': -1.29}
    z_embeds = [[[z1.get(aa, 0.0), z2.get(aa, 0.0), z3.get(aa, 0.0)] for aa in seq] for seq in aa_seqs]
    z_embeds = torch.tensor(z_embeds).float()
    return z_embeds

def collate_fn(examples):
    aa_seqs, names = [], []
    e_descriptor, z_descriptor = [], []
    
    if 'foldseek_seq' in args.structure_seqs:
        foldseek_seqs = []
    if 'esm3_structure_seq' in args.structure_seqs:
        esm3_structure_seqs = []
    
    for e in examples:
        aa_seq = e["aa_seq"]
        aa_seq = re.sub(r"[UZOB]", "X", aa_seq)
        if args.type == "Bacteria":
            aa_seq = " ".join(list(aa_seq))
        else:
            aa_seq = list(aa_seq)
        aa_seqs.append(aa_seq)
        names.append(e.get("name", ""))
        
        if 'foldseek_seq' in args.structure_seqs:
            foldseek_seq = e["foldseek_seq"]
            if args.type == "Bacteria":
                foldseek_seq = " ".join(list(foldseek_seq))
            else:
                foldseek_seq = list(foldseek_seq)
            foldseek_seqs.append(foldseek_seq)
            
        if 'esm3_structure_seq' in args.structure_seqs:
            esm3_structure_seq = [VQVAE_SPECIAL_TOKENS["BOS"]] + e["esm3_structure_seq"] + [VQVAE_SPECIAL_TOKENS["EOS"]]
            esm3_structure_seqs.append(torch.tensor(esm3_structure_seq))
            
        if 'ez_descriptor' in args.structure_seqs:
            e_descriptor.append(torch.tensor(e["e_descriptor"]))
            z_descriptor.append(torch.tensor(e["z_descriptor"]))
    
    if args.type == "Bacteria":
        aa_inputs = tokenizer(aa_seqs, return_tensors="pt", padding=True, truncation=True)
        if 'foldseek_seq' in args.structure_seqs:
            foldseek_input_ids = tokenizer(foldseek_seqs, return_tensors="pt", padding=True, truncation=True)["input_ids"]
    else:
        aa_inputs = tokenizer.batch_encode_plus(aa_seqs, add_special_tokens=True, padding=True, is_split_into_words=True, return_tensors="pt")
        if 'foldseek_seq' in args.structure_seqs:
            foldseek_input_ids = tokenizer.batch_encode_plus(foldseek_seqs, add_special_tokens=True, padding=True, is_split_into_words=True, return_tensors="pt")["input_ids"]
    
    aa_input_ids = aa_inputs["input_ids"]
    attention_mask = aa_inputs["attention_mask"]

    data_dict = {
        "aa_input_ids": aa_input_ids, 
        "attention_mask": attention_mask,
        "names": names
    }

    if 'ez_descriptor' in args.structure_seqs:
        e_descriptor_embeds = torch.stack([torch.cat([e_descriptor[i], torch.zeros(len(aa_input_ids[i]) - len(e_descriptor[i]), 5)], dim=0) for i in range(len(e_descriptor))])
        z_descriptor_embeds = torch.stack([torch.cat([z_descriptor[i], torch.zeros(len(aa_input_ids[i]) - len(z_descriptor[i]), 3)], dim=0) for i in range(len(z_descriptor))])
        data_dict["e_descriptor_embeds"] = e_descriptor_embeds
        data_dict["z_descriptor_embeds"] = z_descriptor_embeds

    if 'foldseek_seq' in args.structure_seqs:
        data_dict["foldseek_input_ids"] = foldseek_input_ids
    if 'esm3_structure_seq' in args.structure_seqs:
        esm3_structure_input_ids = torch.nn.utils.rnn.pad_sequence(
            esm3_structure_seqs, batch_first=True, padding_value=VQVAE_SPECIAL_TOKENS["PAD"]
        )
        if args.type != "Bacteria":
            esm3_structure_input_ids = esm3_structure_input_ids[:,:-1]
        data_dict["esm3_structure_input_ids"] = esm3_structure_input_ids
    
    return data_dict

def infer(model, plm_model, dataloader, device):
    names, pred_labels, pred_probas = [], [], []
    for batch in dataloader:
        names.extend(batch.pop("names"))
        for k, v in batch.items():
            batch[k] = v.to(device)
        
        logits = model(plm_model, batch)
        pred_labels.extend(logits.argmax(dim=1).cpu().numpy())
        pred_probas.extend(logits.softmax(dim=1)[:, 1].cpu().numpy())
    
    return names, pred_labels, pred_probas

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='蛋白质序列分类推理脚本')
    parser.add_argument('-i', '--input', type=str, required=True, help='输入的JSON文件路径')
    parser.add_argument('-t', '--type', type=str, required=True, choices=['Bacteria', 'Virus', 'Tumor'], 
                      help='蛋白质的种属类型')
    parser.add_argument('--structure_seqs', type=str, default='ez_descriptor,foldseek_seq,esm3_structure_seq',
                      help='结构序列类型，用逗号分隔')
    parser.add_argument('--max_seq_len', type=int, default=1024, help='最大序列长度')
    parser.add_argument('--max_batch_token', type=int, default=10000, help='每批次最大token数')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载的工作进程数')
    parser.add_argument('-o', '--output', type=str, help='输出CSV文件路径')
    
    args = parser.parse_args()
    args.structure_seqs = args.structure_seqs.split(',')
    
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 模型参数
    model_params = {
        'hidden_size': None,  # 将由PLM模型自动设置
        'num_attention_heads': 8,
        'attention_probs_dropout_prob': 0,
        'num_labels': 2,
        'pooling_method': 'attention1d',
        'pooling_dropout': 0.25,
        'return_attentions': False,
        'structure_seqs': args.structure_seqs
    }
    
    # 根据种属类型加载不同的PLM模型和分词器
    if args.type == "Bacteria":
        plm_model_name = "Rostlab/prot_bert"
        print(f"正在加载{plm_model_name}模型...")
        tokenizer = BertTokenizer.from_pretrained(plm_model_name, do_lower_case=False)
        plm_model = BertModel.from_pretrained(plm_model_name).to(device).eval()
        model_params['hidden_size'] = plm_model.config.hidden_size
    else:
        plm_model_name = "ElnaggarLab/ankh-large"
        print(f"正在加载{plm_model_name}模型...")
        tokenizer = AutoTokenizer.from_pretrained(plm_model_name, do_lower_case=False)
        plm_model = T5EncoderModel.from_pretrained(plm_model_name).to(device).eval()
        model_params['hidden_size'] = plm_model.config.d_model
    
    # 加载对应种属的模型
    print(f"正在加载{args.type}模型...")
    model = AdapterModel(argparse.Namespace(**model_params))
    model_path = f"ckpt/{args.type}.pt"
    model.load_state_dict(torch.load(model_path))
    model.to(device).eval()
    
    # 处理数据
    print("正在处理输入数据...")
    test_dataset, test_token_num = process_dataset_from_json(args.input, args.max_seq_len)
    test_loader = DataLoader(
        test_dataset,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        batch_sampler=BatchSampler(test_token_num, args.max_batch_token, False)
    )
    
    # 推理
    print("开始推理...")
    with torch.no_grad():
        names, pred_labels, pred_probas = infer(model, plm_model, test_loader, device)
    
    # 保存结果
    output_path = args.output or f"results_{args.type}.csv"
    results_df = pd.DataFrame({
        'name': names,
        'aa_seq': [data['aa_seq'] for data in test_dataset],
        'pred_label': pred_labels,
        'pred_proba': pred_probas
    })
    results_df.to_csv(output_path, index=False)
    print(f"结果已保存到: {output_path}") 