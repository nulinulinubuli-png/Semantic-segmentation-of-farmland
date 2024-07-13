import os
import shutil

import cv2
import torch
from collections import OrderedDict
import glob
import numpy as np
import matplotlib.pyplot as plt

from mypath import Path



label_1_pivix_map = [
    [0, 0, 0],  # 0 未知区域

    [255, 255, 255],  # 1 农业地

]


class Saver(object):

    def __init__(self, args):
        self.args = args
        self.directory = os.path.join('run_helan', args.dataset, args.train_model)
        self.runs = glob.glob(os.path.join(self.directory, 'experiment_*'))
        self.del_path(self.runs)
        self.runs = glob.glob(os.path.join(self.directory, 'experiment_*'))
        run_id = self.save_sorted(glob.glob(os.path.join(self.directory, 'experiment_*')))
        #         run_id = int(self.runs[-1].split('_')[-1]) + 1 if self.runs else 0

        self.experiment_dir = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)))

        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

    def save_sorted(self, runs):
        id = -1
        for run in runs:
            s_id = int(run.split('_')[-1])
            if id < s_id:
                id = s_id
        return id + 1

    def save_loss(self, loss):
        lossfile = os.path.join(self.experiment_dir, self.args.train_model + '-loss.txt')
        loss_file = open(lossfile, 'a+')
        loss_file.write(str(loss) + ",")

    def save_miou(self, miou):
        lossfile = os.path.join(self.experiment_dir, self.args.train_model + '-miou.txt')
        loss_file = open(lossfile, 'a+')
        loss_file.write(str(miou) + ",")

    def del_lossfile(self):
        os.remove(os.path.join(self.experiment_dir, self.args.train_model + '-loss.txt'))

    def del_path(self, runs):
        for run in runs:
            isExist = os.path.exists(os.path.join(run, self.args.train_model + '-loss.txt'))
            if not isExist:
                shutil.rmtree(run)
        return True

    # 保存损失图像
    def save2Png(self, name):
        lossfile = os.path.join(self.experiment_dir, self.args.train_model + "-" + name + '.txt')
        loss_file = open(lossfile, 'r')
        file = loss_file.read().replace(" ", "")
        y_loss = np.asfarray(file[0:-1].split(","), float)
        plt.xlabel("epoch")
        plt.ylabel(name)
        x_loss = range(len(y_loss))
        plt.plot(x_loss, y_loss, linewidth=1, linestyle="solid", label="train loss")
        plt.title(name + " curve: " + self.args.train_model)
        plt.savefig(os.path.join(self.experiment_dir, self.args.train_model + "-" + name + '.jpg'))
        plt.close()

    '''
    对预测结果进行处理，输出为PNG图像
    '''

    def save_images(self, matrix, fname_list, epo):
        n, w, h = matrix.shape
        for k in range(n):
            res = []
            for i in range(w):
                tmp = []
                for j in range(h):
                    if self.args.num_class == 16:
                        if matrix[k, i, j] <= 15:
                            tmp.append(label_1_pivix_map[matrix[k, i, j]])
                        else:
                            tmp.append(label_1_pivix_map[0])
                res.append(tmp)

            res = np.array(res)
            res = np.flip(res, axis=2)
            dst = os.path.join(self.directory, "predict_result")
            if not os.path.exists(dst):
                os.makedirs(dst)
            cv2.imwrite(os.path.join(dst, fname_list[epo * self.args.batch_size + k] + "p.png"), res,
                        [int(cv2.IMWRITE_PNG_COMPRESSION), 2])
            shutil.copyfile(os.path.join(Path.db_root_dir(self.args.dataset), "GTImages",
                                         fname_list[epo * self.args.batch_size + k] + ".tif"),
                            os.path.join(dst, fname_list[epo * self.args.batch_size + k] + "g.tif"))
            shutil.copyfile(os.path.join(Path.db_root_dir(self.args.dataset), "JPEGImages",
                                         fname_list[epo * self.args.batch_size + k] + "_0.tif"),
                            os.path.join(dst, fname_list[epo * self.args.batch_size + k] + ".tif"))


    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        """Saves checkpoint to disk"""
        filename = os.path.join(self.experiment_dir, filename)
        torch.save(state, filename)
        if is_best:
            best_pred = state['best_pred']
            with open(os.path.join(self.experiment_dir, 'best_pred.txt'), 'w') as f:
                f.write(str(best_pred))
            if self.runs:
                previous_miou = [0.0]
                for run in self.runs:
                    run_id = run.split('_')[-1]
                    path = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)), 'best_pred.txt')
                    if os.path.exists(path):
                        with open(path, 'r') as f:
                            miou = float(f.readline())
                            previous_miou.append(miou)
                    else:
                        continue
                max_miou = max(previous_miou)
                if best_pred > max_miou:
                    shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth.tar'))
            else:
                shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth.tar'))

    # 保存parameters文件
    def save_experiment_config(self):
        logfile = os.path.join(self.experiment_dir, 'parameters.txt')
        log_file = open(logfile, 'w')
        p = OrderedDict()
        p['dataset'] = self.args.dataset
        p['backbone'] = self.args.backbone
        p['model'] = self.args.train_model
        p['out_stride'] = self.args.out_stride
        p['lr'] = self.args.lr
        p['lr_scheduler'] = self.args.lr_scheduler
        p['loss_type'] = self.args.loss_type
        p['epoch'] = self.args.epochs
        p['base_size'] = self.args.base_size
        p['crop_size'] = self.args.crop_size
        p['in_channels'] = self.args.in_channels
        p['optimizer'] = self.args.optimizer

        for key, val in p.items():
            log_file.write(key + ':' + str(val) + '\n')
        log_file.close()
