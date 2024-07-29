import numpy as np
import pandas as pd
import shap
import torch

from sklearn.feature_selection import SelectFromModel

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def feature_selection_with_lasso(features_df, labels_df, alpha=0.05, is_shap=None):
    # 合并数据集
    data = pd.merge(features_df, labels_df, on='id')

    # 准备特征和标签
    X = data.drop(columns=['id', data.columns[-1]])  # 假设标签在最后一列
    y = data.iloc[:, -1]

    # 再次检查数据类型，并删除所有非数值列
    non_numeric_columns = X.select_dtypes(exclude=['float64', 'int64']).columns
    if len(non_numeric_columns) > 0:
        print(f"以下列包含非数值数据，将被删除: {non_numeric_columns.tolist()}")
        X.drop(columns=non_numeric_columns, inplace=True)

    # 尝试将所有列转换为浮点类型，如果失败则移除该列
    for column in X.columns:
        try:
            X[column] = X[column].astype(float)
        except ValueError:
            X.drop(column, axis=1, inplace=True)
            print(f"列 '{column}' 已删除，因为包含非数值数据。")

    # 填充缺失值
    X.fillna(X.mean(), inplace=True)

    # 确保特征标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 使用LASSO进行特征选择
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_scaled, y)  # 对Lasso模型进行拟合

    selector = SelectFromModel(estimator=lasso, prefit=True)

    # 筛选出选中的特征及其系数
    selected_features = X.columns[selector.get_support()]
    selected_coefficients = lasso.coef_[selector.get_support()]
    print('Selected features:', selected_features)
    print('Coefficients of selected features:', selected_coefficients)

    # 创建一个DataFrame来存储标准化后的选中特征
    selected_X_scaled = selector.transform(X_scaled)
    selected_features_df = pd.DataFrame(selected_X_scaled, columns=selected_features)

    # 将id列添加到标准化特征DataFrame
    selected_features_df['id'] = data['id'].values

    # 调整列的顺序，将'id'列移至首位
    cols = ['id'] + [col for col in selected_features_df.columns if col != 'id']
    result_df = selected_features_df[cols]

    if is_shap:
        # 使用SHAP解释模型
        explainer = shap.LinearExplainer(lasso, X_scaled, feature_perturbation="interventional")
        shap_values = explainer.shap_values(X_scaled)
        print(selected_features)
        # 仅绘制选中的特征
        shap.summary_plot(shap_values[:, selector.get_support()], X.iloc[:, selector.get_support()],
                          feature_names=selected_features, plot_type="bar", plot_size=(20, 10), show=True)

        # 仅绘制选中的特征
        shap.summary_plot(shap_values[:, selector.get_support()], X.iloc[:, selector.get_support()])




    return result_df

def evaluate_performance(predictions, labels):
    """
    Evaluate the performance of a model based on predictions and actual labels.

    Args:
    predictions (numpy.array): The predicted probabilities for the positive class.
    labels (numpy.array): The true binary labels.

    Returns:
    tuple: A tuple containing the AUC, accuracy, and F1 score.
    """
    # 计算AUC
    predictions = (predictions - np.min(predictions)) / (np.max(predictions) - np.min(predictions))
    val_auc = roc_auc_score(labels, predictions)

    # 计算准确率，首先将概率预测转换为0或1
    pred_labels = (predictions >= 0.5).astype(int)
    acc = accuracy_score(labels, pred_labels)

    # 计算F1分数
    f1 = f1_score(labels, pred_labels)

    return val_auc, acc, f1

def get_cancer(features_list):
    model = torch.load('./is_cancer_20x.pth', map_location=device)
    model.eval()
    output_list = []
    with torch.no_grad():  # 评估模式
        for i in range(0, len(features_list), 2048):
            # 获取当前批次
            batch = features_list[i:i + 2048]
            # 将批次数据转换为张量
            batch_tensor = torch.stack(batch).to(device)
            # 计算模型输出
            output = model(batch_tensor)
            # 将输出添加到输出列表中
            output_list.extend(output.tolist())

    indexed_output = list(enumerate(output_list))
    indexed_output.sort(key=lambda x: x[1], reverse=True)
    sorted_indices = [index for index, value in indexed_output]
    return sorted_indices

def get_longitudinal(data_before_path, data_after_path):
    # 加载数据
    data = pd.read_excel(data_before_path)

    # 按标签值分割为两组
    dataset_bor_0 = data[data['label'] == 0]
    dataset_bor_1 = data[data['label'] == 1]

    # 合并两组数据
    dataset = pd.concat([dataset_bor_0, dataset_bor_1], ignore_index=True)

    # 数据标准化
    scaler = StandardScaler()

    # 提取特征和标签
    X = dataset.loc[:, "original_shape_Elongation":"original_ngtdm_Strength"].values
    X = scaler.fit_transform(X)
    y = dataset['label'].values  # 标签


    # 使用 LASSO 筛选特征
    lasso_model = Lasso(alpha=0.01)
    sfm = SelectFromModel(lasso_model, prefit=False)
    sfm.fit(X, y)

    # 获取选择的特征索引
    selected_feature_indices = sfm.get_support(indices=True)
    dataset_feature = dataset.loc[:, "original_shape_Elongation":"original_ngtdm_Strength"]
    selected_feature_names = dataset_feature.columns[selected_feature_indices].tolist()
    selected_feature_names.append('id')
    selected_feature_names.append('label')
    data_before = dataset.filter(items=selected_feature_names)

    # 加载数据
    data_after = pd.read_excel(data_after_path)

    # 获取选择的特征索引
    selected_feature_indices = sfm.get_support(indices=True)
    dataset_feature = data_after.loc[:, "original_shape_Elongation":"original_ngtdm_Strength"]
    selected_feature_names = dataset_feature.columns[selected_feature_indices].tolist()
    selected_feature_names.append('id')
    data_after = data_after.filter(items=selected_feature_names)

    # 使用 merge 函数进行内连接，保留两个DataFrame中'Patient_number'列相同的行
    merged_inner = pd.merge(data_before, data_after, on='id', how='inner')
    # 提取保留了的'Patient_number'列
    common_patient_numbers = merged_inner['id']
    # 从原始 DataFrame 中筛选保留了'Patient_number'的行
    data_before = data_before[data_before['id'].isin(common_patient_numbers)]
    data_after = data_after[data_after['id'].isin(common_patient_numbers)]
    return data_before, data_after, selected_feature_names
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, my_model, save_optimizer, ckpt_name='checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch, val_loss, my_model, save_optimizer, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(epoch, val_loss, my_model, save_optimizer, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, epoch, val_loss, my_model, save_optimizer, ckpt_name):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # state = {'model': my_model.state_dict(), 'optimizer': save_optimizer.state_dict(), 'epoch': epoch}
        state = my_model.state_dict()
        torch.save(state, ckpt_name)
        # torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss
    def res_best(self):
        x = self.best_score
        return x
