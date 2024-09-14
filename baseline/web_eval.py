import pandas as pd
from sklearn import metrics
from scipy.stats import ks_2samp
import numpy as np
from utils import top_k_accuracy
import matplotlib.pyplot as plt


def evaluate(json_file, xlsx_file, method, fold_num):
    test_exp = pd.read_json(json_file, orient='records', lines=True)
    test_pred = pd.read_excel(xlsx_file, sheet_name=method)
    assert len(test_exp) == len(test_pred)
    df = pd.merge(test_exp, test_pred, left_on='name', right_on='ID')
    metric = []
    for fold in range(1, fold_num + 1):
        df_fold = df.sample(frac=0.5)
        y = df_fold['label']
        pred = df_fold['pred_label']
        pred_proba = df_fold['pred_proba']
        # evaluate
        auc = metrics.roc_auc_score(y, pred)
        accuracy = metrics.accuracy_score(y, pred)
        precision = metrics.precision_score(y, pred)
        recall = metrics.recall_score(y, pred)
        f1 = metrics.f1_score(y, pred)
        mcc = metrics.matthews_corrcoef(y, pred)
        ks, _ = ks_2samp(pred_proba[np.array(y) == 0], pred_proba[np.array(y) == 1])
        entropy = metrics.log_loss(y, pred_proba)
        topk = top_k_accuracy(pred_proba, y)
        metric.append([auc, accuracy, precision, recall, f1, mcc, ks, entropy, topk])
    metric_mean = np.array(metric).mean(axis=0)
    metric_std = np.array(metric).std(axis=0)
    return metric_mean, metric_std


def plot_auc_curve(json_file, xlsx_file, methods, save_fig=None):
    test_exp = pd.read_json(json_file, orient='records', lines=True)
    for method in methods:
        test_pred = pd.read_excel(xlsx_file, sheet_name=method)
        assert len(test_exp) == len(test_pred)
        df = pd.merge(test_exp, test_pred, left_on='name', right_on='ID')
        df_fold = df.sample(frac=0.5)
        y = df_fold['label']
        pred_proba = df_fold['pred_proba']
        fpr, tpr, _ = metrics.roc_curve(y, pred_proba)
        auc = metrics.roc_auc_score(y, pred_proba)
        plt.plot(fpr, tpr, label=f'{method} (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='black', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    if save_fig:
        plt.savefig(save_fig, dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()


def main():
    bacteria_method = ['vaxidl', 'vaxijen2', 'vaxijen3']
    virus_method = ['vaxidl', 'vaxijen2', 'vaxijen3', 'virusimmu']
    tumor_method = ['vaxijen2', 'vaxijen3']

    metric_name = ['AUC', 'Accuracy', 'Precision', 'Recall', 'F1', 'MCC', 'KS-statistic', 'Cross-Entropy', 'TopK']

    # Bacteria
    print('Evaluating Bacteria...')
    metrics_mean, metrics_std = [], []
    for method in bacteria_method:
        metric_mean, metric_std = evaluate('dataset/BacteriaBinary/ESMFold/test.json', 'baseline/data/bacteria_test_prediction_tool.xlsx', method, 10)
        metrics_mean.append(metric_mean)
        metrics_std.append(metric_std)

    df_mean = pd.DataFrame(metrics_mean, columns=metric_name, index=['Vaxi-DL', 'VaxiJen2.0', 'VaxiJen3.0'])
    df_mean.reset_index(inplace=True)
    df_mean.rename(columns={'index': 'Model'}, inplace=True)
    df_std = pd.DataFrame(metrics_std, columns=metric_name, index=['Vaxi-DL', 'VaxiJen2.0', 'VaxiJen3.0'])
    df_std.reset_index(inplace=True)
    df_std.rename(columns={'index': 'Model'}, inplace=True)
    # Save metrics
    prex = f'baseline/results/web_bacteria_metrics'
    df_mean.to_csv(f'{prex}_mean.csv', index=False)
    df_std.to_csv(f'{prex}_std.csv', index=False)
    plot_auc_curve('dataset/BacteriaBinary/ESMFold/test.json', 'baseline/data/bacteria_test_prediction_tool.xlsx', bacteria_method, f'baseline/results/ROC_curve/web_bacteria_ROC_fold1.png')


    # Virus
    print('Evaluating Virus...')
    metrics_mean, metrics_std = [], []
    for method in virus_method:
        metric_mean, metric_std = evaluate('dataset/VirusBinary/ESMFold/test.json', 'baseline/data/virus_test_prediction_tool.xlsx', method, 10)
        metrics_mean.append(metric_mean)
        metrics_std.append(metric_std)
    df_mean = pd.DataFrame(metrics_mean, columns=metric_name, index=['Vaxi-DL', 'VaxiJen2.0', 'VaxiJen3.0', 'VirusImmu'])
    df_mean.reset_index(inplace=True)
    df_mean.rename(columns={'index': 'Model'}, inplace=True)
    df_std = pd.DataFrame(metrics_std, columns=metric_name, index=['Vaxi-DL', 'VaxiJen2.0', 'VaxiJen3.0', 'VirusImmu'])
    df_std.reset_index(inplace=True)
    df_std.rename(columns={'index': 'Model'}, inplace=True)
    # Save metrics
    prex = f'baseline/results/web_virus_metrics'
    df_mean.to_csv(f'{prex}_mean.csv', index=False)
    df_std.to_csv(f'{prex}_std.csv', index=False)
    plot_auc_curve('dataset/VirusBinary/ESMFold/test.json', 'baseline/data/virus_test_prediction_tool.xlsx', virus_method, f'baseline/results/ROC_curve/web_virus_ROC_fold1.png')


    # Tumor
    print('Evaluating Tumor...')
    metrics_mean, metrics_std = [], []
    for method in tumor_method:
        metric_mean, metric_std = evaluate('dataset/TumorBinary/ESMFold/test.json', 'baseline/data/tumor_test_prediction_tool.xlsx', method, 10)
        metrics_mean.append(metric_mean)
        metrics_std.append(metric_std)
    df_mean = pd.DataFrame(metrics_mean, columns=metric_name, index=['VaxiJen2.0', 'VaxiJen3.0'])
    df_mean.reset_index(inplace=True)
    df_mean.rename(columns={'index': 'Model'}, inplace=True)
    df_std = pd.DataFrame(metrics_std, columns=metric_name, index=['VaxiJen2.0', 'VaxiJen3.0'])
    df_std.reset_index(inplace=True)
    df_std.rename(columns={'index': 'Model'}, inplace=True)
    # Save metrics
    prex = f'baseline/results/web_tumor_metrics'
    df_mean.to_csv(f'{prex}_mean.csv', index=False)
    df_std.to_csv(f'{prex}_std.csv', index=False)
    plot_auc_curve('dataset/TumorBinary/ESMFold/test.json', 'baseline/data/tumor_test_prediction_tool.xlsx', tumor_method, f'baseline/results/ROC_curve/web_tumor_ROC_fold1.png')


if __name__ == '__main__':
    main()
