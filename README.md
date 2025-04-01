# VenusVaccine

VenusVaccine is a multimodal deep learning tool for protective antigen prediction, which integrates protein sequences, structures, and physicochemical properties to assist in vaccine candidate selection. The tool can predict whether a protein sequence is likely to be a protective antigen in different pathogen types (Bacteria, Virus, or Tumor).

## Features

- **Multiple Protein Language Models**:
  - Bacteria: Rostlab/prot_bert
  - Virus/Tumor: ElnaggarLab/ankh-large
- **Multimodal Feature Integration**:
  - ez_descriptor: Physicochemical properties (E-descriptor and Z-descriptor)
  - foldseek_seq: Secondary structure sequences from Foldseek
  - esm3_structure_seq: Structure sequences from ESM3

## Installation

```bash
pip install torch transformers pandas numpy tqdm
```

## Usage

### Basic Usage

```bash
python infer.py -i input.json -t Bacteria
```

### Command Line Arguments

```bash
python infer.py [-h] -i INPUT -t {Bacteria,Virus,Tumor} [--structure_seqs STRUCTURE_SEQS] 
                [--max_seq_len MAX_SEQ_LEN] [--max_batch_token MAX_BATCH_TOKEN] 
                [--num_workers NUM_WORKERS] [-o OUTPUT]
```

Arguments:
- `-i, --input`: Path to input JSON file (required)
- `-t, --type`: Pathogen type, choose from: Bacteria, Virus, Tumor (required)
- `--structure_seqs`: Types of structure sequences, comma-separated (default: ez_descriptor,foldseek_seq,esm3_structure_seq)
- `--max_seq_len`: Maximum sequence length (default: 1024)
- `--max_batch_token`: Maximum tokens per batch (default: 10000)
- `--num_workers`: Number of data loading workers (default: 4)
- `-o, --output`: Path to output CSV file (default: results_{type}.csv)

### Input Format

The input should be a JSON file with one sample per line. Fields required depend on the specified structure_seqs parameter:

```json
{
    "name": "protein1",
    "aa_seq": "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
    "foldseek_seq": "HHHEEELLCCHHHHHHHHHHHHSTTHHHHHHHHHHHHHHHHHHHHHHHHEETTEEHHHHHH",
    "esm3_structure_seq": [1, 2, 3, ...],
    "e_descriptor": [[0.1, 0.2, 0.3, 0.4, 0.5], ...],
    "z_descriptor": [[0.1, 0.2, 0.3], ...]
}
```

Required fields:
- `name`: Protein sequence identifier
- `aa_seq`: Amino acid sequence

Optional fields (depending on structure_seqs parameter):
- `foldseek_seq`: Secondary structure sequence predicted by Foldseek
- `esm3_structure_seq`: Structure sequence predicted by ESM3
- `e_descriptor`: E-descriptor features (5-dimensional)
- `z_descriptor`: Z-descriptor features (3-dimensional)

### Output Format

The output is a CSV file containing:
- `name`: Protein sequence identifier
- `aa_seq`: Amino acid sequence
- `pred_label`: Prediction label (0: non-protective antigen, 1: protective antigen)
- `pred_proba`: Prediction probability of being a protective antigen

### Examples

1. Predict using all structural features:
```bash
python infer.py -i proteins.json -t Bacteria
```

2. Use only specific structural features:
```bash
python infer.py -i proteins.json -t Virus --structure_seqs "foldseek_seq,esm3_structure_seq"
```

3. Specify output file:
```bash
python infer.py -i proteins.json -t Tumor -o predictions.csv
```

4. Adjust sequence length and batch size:
```bash
python infer.py -i proteins.json -t Bacteria --max_seq_len 512 --max_batch_token 5000
```

## Important Notes

1. Ensure all required dependencies are installed
2. Make sure corresponding model files exist in the `ckpt` directory (`Bacteria.pt`, `Virus.pt`, or `Tumor.pt`)
3. First run will download pre-trained models automatically (internet connection required)
4. GPU is recommended for better inference performance

## Model Files

Pre-trained model files should be placed in the `ckpt` directory:
- `ckpt/Bacteria.pt`: Model for bacterial protective antigens
- `ckpt/Virus.pt`: Model for viral protective antigens
- `ckpt/Tumor.pt`: Model for tumor protective antigens

## Citation

If you find this tool helpful, please cite our work:
```
@inproceedings{
li2025immunogenicity,
title={Immunogenicity Prediction with Dual Attention Enables Vaccine Target Selection},
author={Song Li and Yang Tan and Song Ke and Liang Hong and Bingxin Zhou},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=hWmwL9gizZ}
}
```
