import argparse
import warnings
import torch
import os
import sys
sys.path.append(os.getcwd())
import wandb
import random
import json
import re
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import logging
from torchmetrics.classification import Accuracy, Recall, Precision, MatthewsCorrCoef, AUROC, F1Score, MatthewsCorrCoef
from torchmetrics.classification import BinaryAccuracy, BinaryRecall, BinaryAUROC, BinaryF1Score, BinaryPrecision, BinaryMatthewsCorrCoef, BinaryF1Score
from torchmetrics.regression import SpearmanCorrCoef
from accelerate import Accelerator
from accelerate.utils import set_seed
from time import strftime, localtime
from datasets import load_dataset
from transformers import EsmTokenizer, EsmModel, BertModel, BertTokenizer
from transformers import T5Tokenizer, T5EncoderModel, AutoTokenizer
from src.utils.data_utils import BatchSampler
from src.models.adapter import AdapterModel
from src.utils.metrics import MultilabelF1Max
from src.utils.loss_fn import MultiClassFocalLossWithAlpha

# ignore warning information
logging.set_verbosity_error()
warnings.filterwarnings("ignore")


def min_max_normalize_dataset(train_dataset, val_dataset, test_dataset):
    labels = [e["label"] for e in train_dataset]
    min_label, max_label = min(labels), max(labels)
    for e in train_dataset:
        e["label"] = (e["label"] - min_label) / (max_label - min_label)
    for e in val_dataset:
        e["label"] = (e["label"] - min_label) / (max_label - min_label)
    for e in test_dataset:
        e["label"] = (e["label"] - min_label) / (max_label - min_label)
    return train_dataset, val_dataset, test_dataset


def train(args, model, plm_model, accelerator, metrics_dict, train_loader, val_loader, test_loader, 
          loss_fn, optimizer, device):
    best_val_loss, best_val_metric_score = float("inf"), -float("inf")
    val_loss_list, val_metric_list = [], []
    path = os.path.join(args.ckpt_dir, args.model_name)
    global_steps = 0
    for epoch in range(args.max_train_epochs):
        print(f"---------- Epoch {epoch} ----------")
        # train
        model.train()
        epoch_train_loss = 0
        epoch_iterator = tqdm(train_loader)
        for batch in epoch_iterator:
            with accelerator.accumulate(model):
                for k, v in batch.items():
                    batch[k] = v.to(device)
                label = batch["label"]
                logits = model(plm_model, batch)
                if args.problem_type == 'regression' and args.num_labels == 1:
                    loss = loss_fn(logits.squeeze(), label.squeeze())
                elif args.problem_type == 'multi_label_classification':
                    loss = loss_fn(logits, label.float())
                else:
                    loss = loss_fn(logits, label)
                epoch_train_loss += loss.item() * len(label)                
                global_steps += 1
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                epoch_iterator.set_postfix(train_loss=loss.item())
                if args.wandb:
                    wandb.log({"train/loss": loss.item(), "train/epoch": epoch}, step=global_steps)
                    
        train_loss = epoch_train_loss / len(train_loader.dataset)
        print(f'EPOCH {epoch} TRAIN loss: {train_loss:.4f}')
        
        # eval every epoch
        model.eval()
        with torch.no_grad():
            val_loss, val_metric_dict = eval_loop(args, model, plm_model, metrics_dict, val_loader, loss_fn, device)
            val_metric_score = val_metric_dict[args.monitor]
            val_metric_list.append(val_metric_score)
            val_loss_list.append(val_loss)
            
            if args.wandb:
                val_log = {"valid/loss": val_loss}
                for metric_name, metric_score in val_metric_dict.items():
                    val_log[f"valid/{metric_name}"] = metric_score
                wandb.log(val_log)
            print(f'EPOCH {epoch} VAL loss: {val_loss:.4f} {args.monitor}: {val_metric_score:.4f}')
    
        # early stopping
        if args.monitor == "loss":
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), path)
                print(f'>>> BEST at epcoh {epoch}, loss: {best_val_loss:.4f}')
                for metric_name, metric_score in val_metric_dict.items():
                    print(f'>>> {metric_name}: {metric_score:.4f}')
                print(f'>>> Save model to {path}')
            
            if len(val_loss_list) - val_loss_list.index(min(val_loss_list)) > args.patience:
                print(f'>>> Early stopping at epoch {epoch}')
                break
        else:
            if val_metric_score > best_val_metric_score:
                best_val_metric_score = val_metric_score
                torch.save(model.state_dict(), path)
                print(f'>>> BEST at epcoh {epoch}, loss: {val_loss:.4f} {args.monitor}: {best_val_metric_score:.4f}')
                for metric_name, metric_score in val_metric_dict.items():
                    print(f'>>> {metric_name}: {metric_score:.4f}')
                print(f'>>> Save model to {path}')
            
            if len(val_metric_list) - val_metric_list.index(max(val_metric_list)) > args.patience:
                print(f'>>> Early stopping at epoch {epoch}')
                break
    
    print(f"TESTING: loading from {path}")
    model.load_state_dict(torch.load(path))
    model.eval()
    with torch.no_grad():
        test_loss, test_metric_dict = eval_loop(args, model, plm_model, metrics_dict, test_loader, loss_fn, device)
        test_metric_score = test_metric_dict[args.monitor]
        
        if args.wandb:
            test_log = {"test/loss": test_loss}
            for metric_name, metric_score in test_metric_dict.items():
                test_log[f"test/{metric_name}"] = metric_score
            wandb.log(test_log)
        print(f'EPOCH {epoch} TEST loss: {test_loss:.4f} {args.monitor}: {test_metric_score:.4f}')
        for metric_name, metric_score in test_metric_dict.items():
            print(f'>>> {metric_name}: {metric_score:.4f}')


def eval_loop(args, model, plm_model, metrics_dict, dataloader, loss_fn, device=None):
    total_loss = 0
    epoch_iterator = tqdm(dataloader)
    
    for batch in epoch_iterator:
        for k, v in batch.items():
            batch[k] = v.to(device)
        label = batch["label"]
        logits = model(plm_model, batch)
        for metric_name, metric in metrics_dict.items():
            if args.problem_type == 'regression' and args.num_labels == 1:
                loss = loss_fn(logits.squeeze(), label.squeeze())
                metric(logits.squeeze(), label.squeeze())
            elif args.problem_type == 'multi_label_classification':
                loss = loss_fn(logits, label.float())
                metric(logits, label)
            else:
                loss = loss_fn(logits, label)
                metric(torch.argmax(logits, 1), label)
        total_loss += loss.item() * len(label)
        epoch_iterator.set_postfix(eval_loss=loss.item())
    
    metrics_result_dict = {}
    epoch_loss = total_loss / len(dataloader.dataset)
    for metric_name, metric in metrics_dict.items():
        metrics_result_dict[metric_name] = metric.compute().item()
        metric.reset()
    return epoch_loss, metrics_result_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # model params
    parser.add_argument('--hidden_size', type=int, default=None, help='embedding hidden size of the model')
    parser.add_argument('--num_attention_heads', type=int, default=8, help='number of attention heads')
    parser.add_argument('--attention_probs_dropout_prob', type=float, default=0, help='attention probs dropout prob')
    parser.add_argument('--plm_model', type=str, default='facebook/esm2_t33_650M_UR50D', help='esm model name')
    parser.add_argument('--pooling_method', type=str, default='attention1d', help='pooling method')
    parser.add_argument('--pooling_dropout', type=float, default=0.25, help='pooling dropout')
    
    # dataset
    parser.add_argument('--dataset', type=str, default=None, help='dataset name')
    parser.add_argument('--dataset_config', type=str, default=None, help='config of dataset')
    parser.add_argument('--num_labels', type=int, default=None, help='number of labels')
    parser.add_argument('--problem_type', type=str, default=None, help='problem type')
    parser.add_argument('--pdb_type', type=str, default=None, choices=[None, 'AlphaFold2', 'ESMFold'], help='pdb type')
    parser.add_argument('--train_file', type=str, default=None, help='train file')
    parser.add_argument('--valid_file', type=str, default=None, help='val file')
    parser.add_argument('--test_file', type=str, default=None, help='test file')
    parser.add_argument('--metrics', type=str, default=None, help='computation metrics')
    
    # train model
    parser.add_argument('--seed', type=int, default=3407, help='random seed')
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='gradient accumulation steps')
    parser.add_argument('--max_batch_token', type=int, default=10000, help='max number of token per batch')
    parser.add_argument('--max_train_epochs', type=int, default=20, help='training epochs')
    parser.add_argument('--max_seq_len', type=int, default=None, help='max sequence length')
    parser.add_argument('--patience', type=int, default=5, help='patience for early stopping')
    parser.add_argument('--monitor', type=str, default=None, help='monitor metric')
    parser.add_argument('--structure_seqs', type=str, default=None, help='structure token')
    parser.add_argument('--loss_fn', type=str, default='cross_entropy', choices=['cross_entropy', 'focal_loss'], help='loss function')
    
    # save model
    parser.add_argument('--model_name', type=str, default=None, help='model name')
    parser.add_argument('--ckpt_root', default="ckpt", help='root directory to save trained models')
    parser.add_argument('--ckpt_dir', default=None, help='directory to save trained models')
    
    # wandb log
    parser.add_argument('--wandb', action='store_true', help='use wandb to log')
    parser.add_argument('--wandb_entity', type=str, default=None, help='wandb entity')
    parser.add_argument('--wandb_project', type=str, default='SES-Adapter')
    parser.add_argument('--wandb_run_name', type=str, default=None)
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.structure_seqs is not None:
        args.structure_seqs = args.structure_seqs.split(',')
        
    dataset_config = json.loads(open(args.dataset_config).read())
    if args.dataset is None:
        args.dataset = dataset_config['dataset']
    if args.pdb_type is None:
        args.pdb_type = dataset_config['pdb_type']
    if args.train_file is None:
        if dataset_config.get('train_file'):
            args.train_file = dataset_config['train_file']
    if args.valid_file is None:
        if dataset_config.get('valid_file'):
            args.valid_file = dataset_config['valid_file']
    if args.test_file is None:
        if dataset_config.get('test_file'):
            args.test_file = dataset_config['test_file']
    if args.num_labels is None:
        args.num_labels = dataset_config['num_labels']
    if args.problem_type is None:
        args.problem_type = dataset_config['problem_type']
    if args.monitor is None:
        args.monitor = dataset_config['monitor']
    
    metrics_dict = {}
    if args.metrics is None:
        args.metrics = dataset_config['metrics']
        if args.metrics != 'None':
            args.metrics = args.metrics.split(',')
            for m in args.metrics:
                if m == 'accuracy':
                    if args.num_labels == 2:
                        metrics_dict[m] = BinaryAccuracy()
                    else:
                        metrics_dict[m] = Accuracy(task="multiclass", num_classes=args.num_labels)
                elif m == 'recall':
                    if args.num_labels == 2:
                        metrics_dict[m] = BinaryRecall()
                    else:
                        metrics_dict[m] = Recall(task="multiclass", num_classes=args.num_labels)
                elif m == 'precision':
                    if args.num_labels == 2:
                        metrics_dict[m] = BinaryPrecision()
                    else:
                        metrics_dict[m] = Precision(task="multiclass", num_classes=args.num_labels)
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
                elif m == 'auc':
                    if args.num_labels == 2:
                        metrics_dict[m] = BinaryAUROC()
                    else:
                        metrics_dict[m] = AUROC(task="multiclass", num_classes=args.num_labels)
                elif m == 'f1_max':
                    metrics_dict[m] = MultilabelF1Max(num_labels=args.num_labels)
                elif m == 'spearman_corr':
                    metrics_dict[m] = SpearmanCorrCoef()
                else:
                    raise ValueError(f"Invalid metric: {m}")
            for metric_name, metric in metrics_dict.items():
                metric.to(device)            
        
    
    # create checkpoint directory
    if args.ckpt_dir is None:
        current_date = strftime("%Y%m%d", localtime())
        args.ckpt_dir = os.path.join(args.ckpt_root, current_date)
    else:
        args.ckpt_dir = os.path.join(args.ckpt_root, args.ckpt_dir)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    
    # init wandb
    if args.wandb:
        if args.wandb_run_name is None:
            args.wandb_run_name = f"Adapter-{args.dataset}"
        if args.model_name is None:
            args.model_name = f"{args.wandb_run_name}.pt"
        
        wandb.init(
            project=args.wandb_project, name=args.wandb_run_name, 
            entity=args.wandb_entity, config=vars(args)
        )
    
    # build tokenizer and protein language model
    if "esm" in args.plm_model:
        print(f"Loading ESM model: {args.plm_model}")
        tokenizer = EsmTokenizer.from_pretrained(args.plm_model)
        plm_model = EsmModel.from_pretrained(args.plm_model).to(device).eval()
        args.hidden_size = plm_model.config.hidden_size
    elif "bert" in args.plm_model:
        print(f"Loading BERT model: {args.plm_model}")
        tokenizer = BertTokenizer.from_pretrained(args.plm_model, do_lower_case=False)
        plm_model = BertModel.from_pretrained(args.plm_model).to(device).eval()
        args.hidden_size = plm_model.config.hidden_size
    elif "prot_t5" in args.plm_model:
        print(f"Loading ProtT5 model: {args.plm_model}")
        tokenizer = T5Tokenizer.from_pretrained(args.plm_model, do_lower_case=False)
        plm_model = T5EncoderModel.from_pretrained(args.plm_model).to(device).eval()
        args.hidden_size = plm_model.config.d_model
    elif "ankh" in args.plm_model:
        print(f"Loading Ankh model: {args.plm_model}")
        tokenizer = AutoTokenizer.from_pretrained(args.plm_model, do_lower_case=False)
        plm_model = T5EncoderModel.from_pretrained(args.plm_model).to(device).eval()
        args.hidden_size = plm_model.config.d_model

    # add 8 dimension for e-descriptor and z-descriptor
    if 'ez_descriptor' in args.structure_seqs:
        args.hidden_size += 8
    elif 'aac' in args.structure_seqs:
        args.hidden_size += 64

    args.vocab_size = plm_model.config.vocab_size

    # load adapter model
    model = AdapterModel(args)
    model.to(device)

    def param_num(model):
        total = sum([param.numel() for param in model.parameters() if param.requires_grad])
        num_M = total/1e6
        if num_M >= 1000:
            return "Number of parameter: %.2fB" % (num_M/1e3)
        else:
            return "Number of parameter: %.2fM" % (num_M)
    print(param_num(model))
    
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
            for seq in args.structure_seqs:
                data[seq] = data[seq][:args.max_seq_len]
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


    if args.train_file is not None and args.train_file[-4:] == 'json':
        train_dataset, train_token_num = process_dataset_from_json(args.train_file)
        val_dataset, val_token_num = process_dataset_from_json(args.valid_file)
        test_dataset, test_token_num = process_dataset_from_json(args.test_file)
    
    # process dataset from list
    def process_dataset_from_list(data_list):
        dataset, token_nums = [], []
        for l in data_list:
            data, token_num = process_data_line(l)
            dataset.append(data)
            token_nums.append(token_num)
        return dataset, token_nums
    
    if args.train_file == None:
        train_dataset, train_token_num = process_dataset_from_list(load_dataset(args.dataset)['train'])
        val_dataset, val_token_num = process_dataset_from_list(load_dataset(args.dataset)['validation'])
        test_dataset, test_token_num = process_dataset_from_list(load_dataset(args.dataset)['test'])
    
    if dataset_config['normalize'] == 'min_max':
        train_dataset, val_dataset, test_dataset = min_max_normalize_dataset(train_dataset, val_dataset, test_dataset)
    
    print(">>> trainset: ", len(train_dataset))
    print(">>> valset: ", len(val_dataset))
    print(">>> testset: ", len(test_dataset))
    print("---------- Smple 3 data point from trainset ----------")
    
    for i in random.sample(range(len(train_dataset)), 3):
        print(">>> ", train_dataset[i])

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

    def aac_embedding(aa_input_ids):
        e_embeds = e_descriptor_embedding(aa_input_ids)
        z_embeds = z_descriptor_embedding(aa_input_ids)
        ez_embeds = torch.cat([e_embeds, z_embeds], dim=-1)
        aac_embeds = []
        for score in ez_embeds:
            aac_scores = []
            k = score.shape[1]
            lag = score.shape[0]
            for l in range(lag):
                aac_score = []
                for i in range(k):
                    for j in range(k):
                        if l == 0:
                            cov = torch.dot(score[:, i], score[:, j]) / lag
                        else:
                            cov = torch.dot(score[:-l, i], score[l:, j]) / (lag - l)
                        aac_score.append(cov.item())
                aac_scores.append(aac_score)
            aac_embeds.append(aac_scores)
        return torch.tensor(aac_embeds).float()

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

        if 'ez_descriptor' in args.structure_seqs:
            # add e-descriptor and z-descriptor embedding
            e_descriptor_embeds = e_descriptor_embedding(aa_input_ids)  # [batch_size, seq_len, 5]
            z_descriptor_embeds = z_descriptor_embedding(aa_input_ids)  # [batch_size, seq_len, 3]
            data_dict["e_descriptor_embeds"] = e_descriptor_embeds
            data_dict["z_descriptor_embeds"] = z_descriptor_embeds

        if 'aac' in args.structure_seqs:
            # add aac embedding
            aac_embeds = aac_embedding(aa_input_ids)
            data_dict["aac_embeds"] = aac_embeds

        if 'foldseek_seq' in args.structure_seqs:
            data_dict["foldseek_input_ids"] = foldseek_input_ids

        return data_dict
        
    # metrics, optimizer, dataloader
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    if args.problem_type == "single_label_classification":
        if args.loss_fn == "cross_entropy":
            loss_fn = nn.CrossEntropyLoss()
        elif args.loss_fn == "focal_loss":
            train_labels = [e["label"] for e in train_dataset]
            alpha = [len(train_labels) / train_labels.count(i) for i in range(args.num_labels)]
            print(">>> alpha: ", alpha)
            loss_fn = MultiClassFocalLossWithAlpha(num_classes=args.num_labels, alpha=alpha, device=device)
    elif args.problem_type == "regression":
        loss_fn = nn.MSELoss()
    elif args.problem_type == "multi_label_classification":
        loss_fn = nn.BCEWithLogitsLoss()
    
    train_loader = DataLoader(
        train_dataset, num_workers=args.num_workers, collate_fn=collate_fn,
        batch_sampler=BatchSampler(train_token_num, args.max_batch_token)
        )
    val_loader = DataLoader(
        val_dataset, num_workers=args.num_workers, collate_fn=collate_fn,
        batch_sampler=BatchSampler(val_token_num, args.max_batch_token, False)
        )
    test_loader = DataLoader(
        test_dataset, num_workers=args.num_workers, collate_fn=collate_fn,
        batch_sampler=BatchSampler(test_token_num, args.max_batch_token, False)
        )
    
    model, optimizer, train_loader, val_loader, test_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader, test_loader
    )
    
    print("---------- Start Training ----------")
    train(args, model, plm_model, accelerator, metrics_dict, train_loader, val_loader, test_loader, loss_fn, optimizer, device)
    
    if args.wandb:
        wandb.finish()