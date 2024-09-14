from torch.utils.data import Dataset
import numpy as np


class MyDataset(Dataset):
    def __init__(self, feature, label):
        assert len(feature) == len(label)
        self.feature = feature
        self.label = label

    def __getitem__(self, index):
        x = self.feature[index]
        y = self.label[index]
        return x, y

    def __len__(self):
        return len(self.feature)


def e_descriptor_score(aa_seqs):
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
    e_scores = [[[e1.get(aa, 0.0), e2.get(aa, 0.0), e3.get(aa, 0.0), e4.get(aa, 0.0), e5.get(aa, 0.0)] for aa in seq] for seq in aa_seqs]
    return e_scores


def z_descriptor_score(aa_seqs):
    z1 = {'A': 0.07, 'R': 2.88, 'N': 3.22, 'D': 3.64, 'C': 0.71, 'Q': 2.18, 'E': 3.08, 'G': 2.23, 'H': 2.41,
          'I': -4.44, 'L': -4.19, 'K': 2.84, 'M': -2.49, 'F': -4.92, 'P': -1.22, 'S': 1.96, 'T': 0.92, 'W': -4.75,
          'Y': -1.39, 'V': -2.69}
    z2 = {'A': -1.73, 'R': 2.52, 'N': 1.45, 'D': 1.13, 'C': -0.97, 'Q': 0.53, 'E': 0.39, 'G': -5.36, 'H': 1.74,
          'I': -1.68, 'L': -1.03, 'K': 1.41, 'M': -0.27, 'F': 1.30, 'P': 0.88, 'S': -1.63, 'T': -2.09, 'W': 3.65,
          'Y': 2.32, 'V': -2.53}
    z3 = {'A': 0.09, 'R': -3.44, 'N': 0.84, 'D': 2.36, 'C': 4.13, 'Q': -1.14, 'E': -0.07, 'G': 0.30, 'H': 1.11,
          'I': -1.03, 'L': -0.98, 'K': -3.14, 'M': -0.41, 'F': 0.45, 'P': 2.23, 'S': 0.57, 'T': -1.40, 'W': 0.85,
          'Y': 0.01, 'V': -1.29}
    z_scores = [[[z1.get(aa, 0.0), z2.get(aa, 0.0), z3.get(aa, 0.0)] for aa in seq] for seq in aa_seqs]
    return z_scores


def get_aac_feature(aa_seqs, descriptor='E', lag=8):

    if min([len(seq) for seq in aa_seqs]) <= lag:
        raise ValueError(f'The shortest sequence in aa_seqs is less than {lag}')

    if descriptor == 'E':
        scores = e_descriptor_score(aa_seqs)
    elif descriptor == 'Z':
        scores = z_descriptor_score(aa_seqs)
    else:
        raise ValueError('Invalid descriptor type')

    aac_scores = []
    for score in scores:
        aac_score = []
        k = len(score[0])
        score = np.array(score)
        for i in range(k):
            for j in range(k):
                for l in range(1, lag+1):
                    cov = np.dot(score[:-l, i], score[l:, j]) / (len(score) - l)
                    aac_score.append(round(cov, 10))
        aac_scores.append(aac_score)
    return aac_scores


def top_k_accuracy(labels, probas):
    probas, labels = np.array(probas), np.array(labels)
    k = int(len(labels) * 0.3)
    topk = probas.argsort()[-k:]
    correct = labels[topk] == 1
    return correct.sum() / k
