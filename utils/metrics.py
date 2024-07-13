import numpy as np
import sklearn
import numpy as np
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt


class Evaluator(object):
    def __init__(self, num_class,args):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)
        self.all_precisions = []
        self.all_recalls = []
        self.args = args

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def IoU1(self, class_index = 1):
        # True Positives (TP): 对角线上的值
        TP = self.confusion_matrix[class_index, class_index]

        # False Positives (FP): 列和中除去TP的部分
        FP = np.sum(self.confusion_matrix[:, class_index]) - TP

        # False Negatives (FN): 行和中除去TP的部分
        FN = np.sum(self.confusion_matrix[class_index, :]) - TP

        # True Negatives (TN): 总和中除去TP, FP, FN的部分
        TN = np.sum(self.confusion_matrix) - TP - FP - FN

        # IoU 计算
        IoU = TP / (TP + FP + FN)

        return IoU
    # def PR_curve

    def IoU0(self, class_index = 0):
        # True Positives (TP): 对角线上的值
        TP = self.confusion_matrix[class_index, class_index]

        # False Positives (FP): 列和中除去TP的部分
        FP = np.sum(self.confusion_matrix[:, class_index]) - TP

        # False Negatives (FN): 行和中除去TP的部分
        FN = np.sum(self.confusion_matrix[class_index, :]) - TP

        # True Negatives (TN): 总和中除去TP, FP, FN的部分
        TN = np.sum(self.confusion_matrix) - TP - FP - FN

        # IoU 计算
        IoU = TP / (TP + FP + FN)

        return IoU

    def Overall_Accuracy(self):
        correct_classified = np.sum(np.diag(self.confusion_matrix))
        total_pixels = np.sum(self.confusion_matrix)
        overall_accuracy = correct_classified / total_pixels
        return overall_accuracy

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU
    
    def Mean_Intersection_over_Union_Class(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        return MIoU


    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def Kappa(self):
        OA = self.Pixel_Accuracy()
        KC = np.matmul(np.sum(self.confusion_matrix,axis=0),np.sum(self.confusion_matrix,axis=1)) / np.sum(self.confusion_matrix)**2
        return (OA - KC) / (1 - KC)
    def calculate_f1_score(self):
        num_classes = self.confusion_matrix.shape[0]
        f1_scores = []

        for class_idx in range(num_classes):
            # 计算精确率（Precision）
            precision = self.confusion_matrix[class_idx, class_idx] / np.sum(self.confusion_matrix[:, class_idx])

            # 计算召回率（Recall）
            recall = self.confusion_matrix[class_idx, class_idx] / np.sum(self.confusion_matrix[class_idx, :])

            # 计算 F1 分数
            if precision + recall == 0:
                f1 = 0  # 避免除以零错误
            else:
                f1 = 2 * (precision * recall) / (precision + recall)

            f1_scores.append(f1)

        # 计算平均 F1 分数
        mean_f1_score = np.mean(f1_scores)

        return mean_f1_score,f1_scores

    def cm2F1(self):
        hist = self.confusion_matrix
        n_class = hist.shape[0]
        tp = np.diag(hist)
        sum_a1 = hist.sum(axis=1)
        sum_a0 = hist.sum(axis=0)
        # ---------------------------------------------------------------------- #
        # 1. Accuracy & Class Accuracy
        # ---------------------------------------------------------------------- #
        acc = tp.sum() / (hist.sum() + np.finfo(np.float32).eps)

        # recall
        recall = tp / (sum_a1 + np.finfo(np.float32).eps)
        # acc_cls = np.nanmean(recall)

        # precision
        precision = tp / (sum_a0 + np.finfo(np.float32).eps)

        # F1 score
        F1 = 2 * recall * precision / (recall + precision + np.finfo(np.float32).eps)
        mean_F1 = np.nanmean(F1)
        return mean_F1

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)




