import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import numpy as np
from sklearn import metrics
from scipy.stats import ks_2samp
from utils import get_aac_feature, top_k_accuracy
import argparse
import json
import warnings
warnings.filterwarnings("ignore")


class MLModel(object):
    def __init__(self, train_data, test_data, args):
        self.x_train = get_aac_feature(train_data['aa_seq'].tolist(), args.descriptor, args.lag)
        self.y_train = train_data['label'].tolist()
        self.x_test = get_aac_feature(test_data['aa_seq'].tolist(), args.descriptor, args.lag)
        self.y_test = test_data['label'].tolist()

    def model(self, model_name):
        rs = 2024
        # ensemble learning - bagging: random forest
        if model_name == 'rf':
            return RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=rs)
        # ensemble learning - boosting: decision tree, gradient boosting, adaptive boosting, xgboost
        elif model_name == 'dt':
            return tree.DecisionTreeClassifier(max_depth=3, random_state=rs)
        elif model_name == 'gbdt':
            return GradientBoostingClassifier(n_estimators=100, random_state=rs)
        elif model_name == 'abdt':
            return AdaBoostClassifier(n_estimators=100, random_state=rs)
        elif model_name == 'xgb':
            return XGBClassifier(n_estimators=100, random_state=rs)
        # optimization algorithm: stochastic gradient descent, logistic regression
        elif model_name == 'sgd':
            return SGDClassifier(loss='modified_huber', max_iter=100, n_jobs=-1, random_state=rs)
        elif model_name == 'lr':
            return LogisticRegression(solver='liblinear', max_iter=100, n_jobs=-1, random_state=rs)
        # neural network: multi-layer perceptron
        elif model_name == 'mlp':
            return MLPClassifier(hidden_layer_sizes=(128, 64), solver='adam', activation='relu', max_iter=100, early_stopping=True, random_state=rs)
        # support vector machine
        elif model_name == 'svm':
            return SVC(max_iter=100, probability=True, random_state=rs)
        # k-nearest neighbors
        elif model_name == 'knn':
            return KNeighborsClassifier(n_neighbors=2, n_jobs=-1)
        elif model_name == 'vote':
            xgb = XGBClassifier(n_estimators=100, random_state=rs)
            rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=rs)
            knn = KNeighborsClassifier(n_neighbors=2, n_jobs=-1)
            return VotingClassifier(estimators=[('rf_', rf), ('xgb_', xgb), ('knn_', knn)], voting='soft')
        elif model_name == 'virusimmu':
            return [self.model('xgb'), self.model('rf'), self.model('knn')]
        elif model_name == 'all':
            return [self.model('rf'), self.model('gbdt'), self.model('xgb'), self.model('sgd'), self.model('lr'), self.model('mlp'), self.model('svm'), self.model('knn')]
        else:
            raise ValueError('Invalid model name')

    def predict_proba(self, model_name):
        if model_name == 'all':
            preds = []
            for model in self.model(model_name):
                model.fit(self.x_train, self.y_train)
                pred = model.predict_proba(self.x_test)[:, 1]
                preds.append(pred)
            return preds
        elif model_name == 'virusimmu':
            weight = [0.05, 0.75, 0.2]
            preds = []
            for model in self.model(model_name):
                model.fit(self.x_train, self.y_train)
                pred = model.predict_proba(self.x_test)[:, 1]
                preds.append(pred)
            pred = weight[0] * preds[0] + weight[1] * preds[1] + weight[2] * preds[2]
        else:
            model = self.model(model_name)
            model.fit(self.x_train, self.y_train)
            pred = model.predict_proba(self.x_test)[:, 1]
        return pred

    def predict(self, model_name):
        if model_name == 'all':
            preds = []
            for model in self.model(model_name):
                model.fit(self.x_train, self.y_train)
                pred = model.predict(self.x_test)
                preds.append(pred)
            return preds
        elif model_name == 'virusimmu':
            pred_proba = self.predict_proba(model_name)
            pred = [1 if p > 0.5 else 0 for p in pred_proba]
        else:
            model = self.model(model_name)
            model.fit(self.x_train, self.y_train)
            pred = model.predict(self.x_test)
        return pred

    # evaluate the performance(AUC, Accuracy, ...) of the model
    def evaluate(self, model_name):
        assert model_name not in ['all'], 'Please use evaluate_all() to evaluate all models.'
        pred_proba = self.predict_proba(model_name)
        pred = self.predict(model_name)
        auc = metrics.roc_auc_score(self.y_test, pred_proba)
        accuracy = metrics.accuracy_score(self.y_test, pred)
        precision = metrics.precision_score(self.y_test, pred)
        recall = metrics.recall_score(self.y_test, pred)
        f1 = metrics.f1_score(self.y_test, pred)
        mcc = metrics.matthews_corrcoef(self.y_test, pred)
        ks = ks_2samp(pred_proba[np.array(self.y_test) == 0], pred_proba[np.array(self.y_test) == 1]).statistic
        entropy = metrics.log_loss(self.y_test, pred_proba)
        topk = top_k_accuracy(self.y_test, pred_proba)
        return auc, accuracy, precision, recall, f1, mcc, ks, entropy, topk

    def evaluate_all(self):
        results = []
        for model_name in ['rf', 'gbdt', 'xgb', 'sgd', 'lr', 'mlp', 'svm', 'knn']:
            auc, accuracy, precision, recall, f1, mcc, ks, entropy, topk = self.evaluate(model_name)
            results.append([auc, accuracy, precision, recall, f1, mcc, ks, entropy, topk])
        return results

    def plot_auc_curve(self, model_name, save_fig=None):
        if model_name == 'all':
            # plot ROC curve for all models
            for model_name in ['rf', 'gbdt', 'xgb', 'sgd', 'lr', 'mlp', 'svm', 'knn']:
                pred_proba = self.predict_proba(model_name)
                fpr, tpr, _ = metrics.roc_curve(self.y_test, pred_proba)
                auc = metrics.roc_auc_score(self.y_test, pred_proba)
                plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.2f})')
        else:
            pred_proba = self.predict_proba(model_name)
            fpr, tpr, _ = metrics.roc_curve(self.y_test, pred_proba)
            auc = metrics.roc_auc_score(self.y_test, pred_proba)
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve')
        plt.legend()
        if save_fig:
            plt.savefig(save_fig, dpi=300, bbox_inches='tight')
        # plt.show()
        plt.close()

    def plot_proba(self, model_name):
        pred_proba = self.predict_proba(model_name)
        plt.hist(pred_proba[np.array(self.y_test) == 0], bins=100, color='b', alpha=0.5, label='0')
        plt.hist(pred_proba[np.array(self.y_test) == 1], bins=100, color='r', alpha=0.5, label='1')
        plt.xlabel('Probability')
        plt.ylabel('Frequency')
        plt.title(f'Probability Distribution of {model_name}')
        plt.legend()
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', '-m', type=str, default='all')
    parser.add_argument('--descriptor', '-d', type=str, default='E', choices=['E', 'Z'])
    parser.add_argument('--lag', '-l', type=int, default=8)
    parser.add_argument('--fold', '-f', type=int, default=1)
    parser.add_argument('--species', '-s', type=str, default='bacteria', choices=['bacteria', 'virus', 'tumor'])
    parser.add_argument('--output_dir', '-o', type=str, default='baseline/results')
    parser.add_argument('--save_result', '-sr', action='store_true')
    parser.add_argument('--plot_auc', '-pa', action='store_true')
    parser.add_argument('--plot_proba', '-pp', action='store_true')
    args = parser.parse_args()

    config_file = f'dataset/{args.species.capitalize()}Binary/{args.species.capitalize()}Binary_ESMFold.json'
    config = json.load(open(config_file, 'r'))
    train_file = config['train_file']
    test_file = config['test_file']
    train_data = pd.read_json(train_file, orient='records', lines=True)
    test_data_all = pd.read_json(test_file, orient='records', lines=True)

    metric_name = ['AUC', 'Accuracy', 'Precision', 'Recall', 'F1', 'MCC', 'KS-statistic', 'Cross-Entropy', 'TopK']
    metrics_all = []
    for fold in range(1, args.fold + 1):
        test_data = test_data_all.sample(frac=0.5)
        print('='*30 + f"Fold {fold}: {len(test_data)} data for evaluation" + '='*30)

        model = MLModel(train_data, test_data, args)
        if args.model_name == 'all':
            model_name = ['Random Forest', 'Gradient Boosting', 'XGBoost', 'SGD', 'Logistic Regression', 'MLP', 'SVM', 'KNN']
            results = model.evaluate_all()
            metrics_all.append(results)
        else:
            model_name = [args.model_name]
            auc, accuracy, precision, recall, f1, mcc, ks, loss, topk = model.evaluate(model_name=args.model_name)
            metrics_all.append([[auc, accuracy, precision, recall, f1, mcc, ks, loss, topk]])
        df_temp = pd.DataFrame(metrics_all[-1], columns=metric_name, index=model_name)
        print(df_temp)

        if args.plot_auc:
            fig_dir = f'{args.output_dir}/ROC_curve'
            model.plot_auc_curve(model_name=args.model_name, save_fig=f'{fig_dir}/{args.model_name}_{args.species}_ROC_fold{fold}.png')

        if args.plot_proba:
            model.plot_proba(model_name=args.model_name)

    metrics_all = np.array(metrics_all)
    metrics_mean = np.mean(metrics_all, axis=0)
    metrics_std = np.std(metrics_all, axis=0)
    df_mean = pd.DataFrame(metrics_mean, columns=metric_name, index=model_name)
    df_mean.reset_index(inplace=True)
    df_mean.rename(columns={'index': 'Model'}, inplace=True)
    df_std = pd.DataFrame(metrics_std, columns=metric_name, index=model_name)
    df_std.reset_index(inplace=True)
    df_std.rename(columns={'index': 'Model'}, inplace=True)

    if args.save_result:
        prex = f'{args.output_dir}/{args.model_name}_{args.species}_metrics'
        df_mean.to_csv(f'{prex}_mean.csv', index=False)
        df_std.to_csv(f'{prex}_std.csv', index=False)
    print('='*30 + 'Mean Metrics' + '='*30)
    print(df_mean)
    print('='*30 + 'Standard Deviation' + '='*30)
    print(df_std)


if __name__ == '__main__':
    main()
