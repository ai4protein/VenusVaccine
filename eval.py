import argparse
import numpy as np
import torch
import re
import json
import os
import warnings
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from torchmetrics.classification import Accuracy, Recall, Precision, MatthewsCorrCoef, AUROC, F1Score
from torchmetrics.classification import BinaryAUROC, BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryMatthewsCorrCoef
from torchmetrics.regression import SpearmanCorrCoef
from transformers import EsmTokenizer, EsmModel, BertModel, BertTokenizer
from transformers import T5Tokenizer, T5EncoderModel, AutoTokenizer
from transformers import logging
from datasets import load_dataset
from torch.utils.data import DataLoader
from src.utils.data_utils import BatchSampler
from src.utils.data_utils import top_k_accuracy, plot_roc_curve
from src.utils.metrics import MultilabelF1Max
from src.models.adapter import AdapterModel
from scipy.stats import ks_2samp
from sklearn import metrics

# ignore warning information
logging.set_verbosity_error()
warnings.filterwarnings("ignore")


def evaluate(model, plm_model, dataloader, device=None, plot_auc=False, fold_num=10, args=None):
    epoch_iterator = tqdm(dataloader)
    labels, pred_labels, pred_probas = [], [], []
    for batch in epoch_iterator:
        for k, v in batch.items():
            batch[k] = v.to(device)
        logits = model(plm_model, batch)
        labels.extend(batch["label"].cpu().numpy())
        pred_labels.extend(logits.argmax(dim=1).cpu().numpy())
        pred_probas.extend(logits.softmax(dim=1)[:, 1].cpu().numpy())

    test_data_all = pd.DataFrame({'label': labels, 'pred_label': pred_labels, 'pred_proba': pred_probas})
    metric_name = ['AUC', 'Accuracy', 'Precision', 'Recall', 'F1', 'MCC', 'KS-statistic', 'Cross-Entropy', 'TopK']
    metrics_all = []
    for fold in range(1, fold_num + 1):
        test_data = test_data_all.sample(frac=0.5)
        print('=' * 30 + f"Fold {fold}: {len(test_data)} data for evaluation" + '=' * 30)
        auc = metrics.roc_auc_score(test_data['label'], test_data['pred_proba'])
        accuracy = metrics.accuracy_score(test_data['label'], test_data['pred_label'])
        precision = metrics.precision_score(test_data['label'], test_data['pred_label'])
        recall = metrics.recall_score(test_data['label'], test_data['pred_label'])
        f1 = metrics.f1_score(test_data['label'], test_data['pred_label'])
        mcc = metrics.matthews_corrcoef(test_data['label'], test_data['pred_label'])
        ks = ks_2samp(test_data[test_data['label'] == 1]['pred_proba'], test_data[test_data['label'] == 0]['pred_proba']).statistic
        entropy = metrics.log_loss(test_data['label'], test_data['pred_proba'])
        topk = top_k_accuracy(test_data['label'], test_data['pred_proba'])

        metrics_all.append([[auc, accuracy, precision, recall, f1, mcc, ks, entropy, topk]])
        df_temp = pd.DataFrame(metrics_all[-1], columns=metric_name, index=[f'Fold{fold}'])
        print(df_temp)

        if plot_auc and fold == 1:
            plot_roc_curve(test_data['label'], test_data['pred_proba'], save_fig=f"{args.test_result_dir}/{args.model_name.split('.')[0]}_roc_curve.png")
            print(f"ROC curve saved in {args.test_result_dir}/roc_curve.png")

    metrics_all = np.array(metrics_all)
    metrics_mean = np.mean(metrics_all, axis=0)
    metrics_std = np.std(metrics_all, axis=0)
    df_mean = pd.DataFrame(metrics_mean, columns=metric_name, index=[args.model_name.split('.')[0]])
    df_mean.reset_index(inplace=True)
    df_mean.rename(columns={'index': 'Model'}, inplace=True)
    df_std = pd.DataFrame(metrics_std, columns=metric_name, index=[args.model_name.split('.')[0]])
    df_std.reset_index(inplace=True)
    df_std.rename(columns={'index': 'Model'}, inplace=True)

    if args.test_result_dir:
        prex = f"{args.test_result_dir}/{args.model_name.split('.')[0]}"
        df_mean.to_csv(f'{prex}_mean.csv', index=False)
        df_std.to_csv(f'{prex}_std.csv', index=False)
    print('=' * 30 + 'Mean Metrics' + '=' * 30)
    print(df_mean)
    print('=' * 30 + 'Standard Deviation' + '=' * 30)
    print(df_std)
    return metrics_mean[0], pred_labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model params
    parser.add_argument('--hidden_size', type=int, default=None, help='embedding hidden size of the model')
    parser.add_argument('--num_attention_heads', type=int, default=8, help='number of attention heads')
    parser.add_argument('--attention_probs_dropout_prob', type=float, default=0, help='attention probs dropout prob')
    parser.add_argument('--plm_model', type=str, default='facebook/esm2_t33_650M_UR50D', help='esm model name')
    parser.add_argument('--num_labels', type=int, default=2, help='number of labels')
    parser.add_argument('--pooling_method', type=str, default='attention1d', help='pooling method')
    parser.add_argument('--pooling_dropout', type=float, default=0.25, help='pooling dropout')
    
    # dataset
    parser.add_argument('--dataset', type=str, default=None, help='dataset name')
    parser.add_argument('--problem_type', type=str, default=None, help='problem type')
    parser.add_argument('--test_file', type=str, default=None, help='test file')
    parser.add_argument('--test_result_dir', type=str, default=None, help='test result directory')
    parser.add_argument('--metrics', type=str, default=None, help='computation metrics')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    parser.add_argument('--max_seq_len', type=int, default=None, help='max sequence length')
    parser.add_argument('--max_batch_token', type=int, default=10000, help='max number of token per batch')
    parser.add_argument('--structure_seqs', type=str, default=None, help='structure token')
    
    # model path
    parser.add_argument('--model_name', type=str, default=None, help='model name')
    parser.add_argument('--ckpt_root', default="result", help='root directory to save trained models')
    parser.add_argument('--ckpt_dir', default=None, help='directory to save trained models')
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.test_result_dir, exist_ok=True)
    # build tokenizer and protein language model
    if "esm" in args.plm_model:
        tokenizer = EsmTokenizer.from_pretrained(args.plm_model)
        plm_model = EsmModel.from_pretrained(args.plm_model).to(device).eval()
        args.hidden_size = plm_model.config.hidden_size
    elif "bert" in args.plm_model:
        tokenizer = BertTokenizer.from_pretrained(args.plm_model, do_lower_case=False)
        plm_model = BertModel.from_pretrained(args.plm_model).to(device).eval()
        args.hidden_size = plm_model.config.hidden_size
    elif "prot_t5" in args.plm_model:
        tokenizer = T5Tokenizer.from_pretrained(args.plm_model, do_lower_case=False)
        plm_model = T5EncoderModel.from_pretrained(args.plm_model).to(device).eval()
        args.hidden_size = plm_model.config.d_model
    elif "ankh" in args.plm_model:
        tokenizer = AutoTokenizer.from_pretrained(args.plm_model, do_lower_case=False)
        plm_model = T5EncoderModel.from_pretrained(args.plm_model).to(device).eval()
        args.hidden_size = plm_model.config.d_model
    # add 8 dimension for e-descriptor and z-descriptor
    if 'ez_descriptor' in args.structure_seqs:
        args.hidden_size += 8
    elif 'aac' in args.structure_seqs:
        args.hidden_size += 64
    args.vocab_size = plm_model.config.vocab_size
    
    metrics_dict = {}
    args.metrics = args.metrics.split(',')
    for m in args.metrics:
        if m == 'auc':
            if args.num_labels == 2:
                metrics_dict[m] = BinaryAUROC()
            else:
                metrics_dict[m] = AUROC(task="multiclass", num_classes=args.num_labels)
        elif m == 'accuracy':
            if args.num_labels == 2:
                metrics_dict[m] = BinaryAccuracy()
            else:
                metrics_dict[m] = Accuracy(task="multiclass", num_classes=args.num_labels)
        elif m == 'precision':
            if args.num_labels == 2:
                metrics_dict[m] = BinaryPrecision()
            else:
                metrics_dict[m] = Precision(task="multiclass", num_classes=args.num_labels)
        elif m == 'recall':
            if args.num_labels == 2:
                metrics_dict[m] = BinaryRecall()
            else:
                metrics_dict[m] = Recall(task="multiclass", num_classes=args.num_labels)
        elif m == 'f1':
            if args.num_labels == 2:
                metrics_dict[m] = BinaryF1Score()
            else:
                metrics_dict[m] = F1Score(task="multiclass", num_classes=args.num_labels)
        elif m == 'mcc':
            if args.num_labels == 2:
                metrics_dict[m] = BinaryMatthewsCorrCoef()
            else:
                metrics_dict[m] = MatthewsCorrCoef(task="multiclass", num_classes=args.num_labels)
        elif m == 'f1_max':
            metrics_dict[m] = MultilabelF1Max(num_labels=args.num_labels)
        elif m == 'spearman_corr':
            metrics_dict[m] = SpearmanCorrCoef()
        else:
            raise ValueError(f"Invalid metric: {m}")
    for metric_name, metric in metrics_dict.items():
        metric.to(device)     
    
    
    # load adapter model
    print("---------- Load Model ----------")
    model = AdapterModel(args)
    model_path = f"{args.ckpt_root}/{args.ckpt_dir}/{args.model_name}"
    model.load_state_dict(torch.load(model_path))
    model.to(device).eval()

    def param_num(model):
        total = sum([param.numel() for param in model.parameters() if param.requires_grad])
        num_M = total/1e6
        if num_M >= 1000:
            return "Number of parameter: %.2fB" % (num_M/1e3)
        else:
            return "Number of parameter: %.2fM" % (num_M)
    print(param_num(model))
     

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
        aa_seqs, labels = [], []
        if 'foldseek_seq' in args.structure_seqs:
            foldseek_seqs = []
        for e in examples:
            aa_seq = e["aa_seq"]
            if 'foldseek_seq' in args.structure_seqs:
                foldseek_seq = e["foldseek_seq"]
            
            if 'prot_bert' in args.plm_model or "prot_t5" in args.plm_model:
                aa_seq = " ".join(list(aa_seq))
                aa_seq = re.sub(r"[UZOB]", "X", aa_seq)
                if 'foldseek_seq' in args.structure_seqs:
                    foldseek_seq = " ".join(list(foldseek_seq))
            elif 'ankh' in args.plm_model:
                aa_seq = list(aa_seq)
                if 'foldseek_seq' in args.structure_seqs:
                    foldseek_seq = list(foldseek_seq)
            
            aa_seqs.append(aa_seq)
            if 'foldseek_seq' in args.structure_seqs:
                foldseek_seqs.append(foldseek_seq)
            labels.append(e["label"])
        
        if 'ankh' in args.plm_model:
            aa_inputs = tokenizer.batch_encode_plus(aa_seqs, add_special_tokens=True, padding=True, is_split_into_words=True, return_tensors="pt")
            if 'foldseek_seq' in args.structure_seqs:
                foldseek_input_ids = tokenizer.batch_encode_plus(foldseek_seqs, add_special_tokens=True, padding=True, is_split_into_words=True, return_tensors="pt")["input_ids"]
        else:
            aa_inputs = tokenizer(aa_seqs, return_tensors="pt", padding=True, truncation=True)
            if 'foldseek_seq' in args.structure_seqs:
                foldseek_input_ids = tokenizer(foldseek_seqs, return_tensors="pt", padding=True, truncation=True)["input_ids"]
        
        aa_input_ids = aa_inputs["input_ids"]
        attention_mask = aa_inputs["attention_mask"]
        
        if args.problem_type == 'regression':
            labels = torch.as_tensor(labels, dtype=torch.float)
        else:
            labels = torch.as_tensor(labels, dtype=torch.long)
        
        data_dict = {"aa_input_ids": aa_input_ids, "attention_mask": attention_mask, "label": labels}
        if 'foldseek_seq' in args.structure_seqs:
            data_dict["foldseek_input_ids"] = foldseek_input_ids
        if 'ez_descriptor' in args.structure_seqs:
            # add e-descriptor and z-descriptor embedding
            e_descriptor_embeds = e_descriptor_embedding(aa_input_ids)  # [batch_size, seq_len, 5]
            z_descriptor_embeds = z_descriptor_embedding(aa_input_ids)  # [batch_size, seq_len, 3]
            data_dict["e_descriptor_embeds"] = e_descriptor_embeds
            data_dict["z_descriptor_embeds"] = z_descriptor_embeds
        return data_dict
        
    loss_fn = nn.CrossEntropyLoss()
    
    def process_data_line(data):
        if args.problem_type == 'multi_label_classification':
            label_list = data['label'].split(',')
            data['label'] = [int(l) for l in label_list]
            binary_list = [0] * args.num_labels
            for index in data['label']:
                binary_list[index] = 1
            data['label'] = binary_list
        if args.max_seq_len is not None:
            data["aa_seq"] = data["aa_seq"][:args.max_seq_len]
            if 'foldseek_seq' in args.structure_seqs:
                data["foldseek_seq"] = data["foldseek_seq"][:args.max_seq_len]
            token_num = min(len(data["aa_seq"]), args.max_seq_len)
        else:
            token_num = len(data["aa_seq"])
        return data, token_num
    
    # process dataset from json file
    def process_dataset_from_json(file):
        dataset, token_nums = [], []
        for l in open(file):
            data = json.loads(l)
            data, token_num = process_data_line(data)
            dataset.append(data)
            token_nums.append(token_num)
        return dataset, token_nums


    # process dataset from list
    def process_dataset_from_list(data_list):
        dataset, token_nums = [], []
        for l in data_list:
            data, token_num = process_data_line(l)
            dataset.append(data)
            token_nums.append(token_num)
        return dataset, token_nums
    
    
    if args.test_file.endswith('json'):
        test_dataset, test_token_num = process_dataset_from_json(args.test_file)
    elif args.test_file.endswith('csv'):
        test_dataset, test_token_num = process_dataset_from_list(load_dataset("csv", data_files=args.test_file)['train'])
        if args.test_result_dir:
            test_result_df = pd.read_csv(args.test_file)
    else:
        raise ValueError("Invalid file format")
    
        
    test_loader = DataLoader(
        test_dataset, num_workers=args.num_workers, collate_fn=collate_fn,
        batch_sampler=BatchSampler(test_token_num, args.max_batch_token, False)
        )

    print("---------- Start Eval ----------")
    with torch.no_grad():
        result, pred_labels = evaluate(model, plm_model, test_loader, device, plot_auc=True, args=args)
