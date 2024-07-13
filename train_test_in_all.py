import argparse
import os
from tqdm import tqdm
import torch
import numpy as np
import torch.nn.functional as F

from utils.adan import Adan

from mypath import Path
from dataloaders import make_data_loader
from modeling.sync_batchnorm.replicate import patch_replication_callback

from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator


from modeling.IAIPENet import IAIPENet as MFEPNet
import time

# Builder类负责根据命令行参数选择和构建适当的图像分割模型。
# 支持多种不同的模型架构，包括MSCSANet、IMSCSANet、SwinUperNet等。
class Builder(object):
    def __init__(self, args):
        self.args = args
        self.models = {

            'MFEPNet':MFEPNet,

        }

    # 该方法的目的是根据命令行提供的train_model参数值构建并返回一个神经网络模型。
    def build_model(self):
        if self.args.train_model == None or self.args.train_model not in self.models:
            raise NotImplementedError
        # print(self.args.GA_Stages)
        model = self.models[self.args.train_model]
        if model in (MFEPNet,):
            return model(num_classes=self.args.num_class, pretrain_img_size=self.args.base_size,
                         in_chans=self.args.in_channels, use_attens=self.args.use_attens)


class Trainer(object):

    def __init__(self, args):
        self.args = args

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()
        
        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        # 创建数据加载器，分别用于训练、验证和测试
        # self.nclass代表数据集中的类别数量
        # self.file_list是数据加载器返回的文件列表，包含了数据集中每个样本的文件路径信息。
        # 这个列表可能在训练过程中用于保存模型的预测结果或其他用途。
        # 数据加载器
        self.train_loader, self.val_loader, self.test_loader, self.nclass, self.file_list = make_data_loader(args, **kwargs)

        # Define network
        model = Builder(args = self.args).build_model()
        # 定义了训练参数，包括要训练的模型参数和学习率。
        # model.parameters() 返回一个包含了模型所有可训练参数的生成器或迭代器。
        train_params = [{'params': model.parameters(), 'lr': args.lr}]
        # train_params = [{'params': model.parameters()}]

        # Define Optimizer
        if args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(train_params, momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=args.nesterov, lr=args.lr)
        elif args.optimizer == 'AdamW':
            optimizer = torch.optim.AdamW(train_params, weight_decay = args.weight_decay)
        elif args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(train_params, weight_decay=args.weight_decay)
        elif args.optimizer == 'Adan':
            optimizer = Adan(train_params, weight_decay=args.weight_decay, betas=args.opt_betas,
                             eps = args.opt_eps, max_grad_norm=args.max_grad_norm, no_prox=args.no_prox)

        # Define Criterion(损失函数)
        # whether to use class balanced weights
        #  args.use_balanced_weights 为 False，或者类别权重已经加载，
        #  就将权重存储在名为 weight 的变量中。这个权重是一个包含类别权重的张量（tensor）。
        if args.use_balanced_weights:
            classes_weights_path = os.path.join(Path.db_root_dir(args.dataset), args.dataset+'_classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(args.dataset, self.train_loader, self.nclass)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None
        # 使用 SegmentationLosses 类来构建损失函数，通过传递权重 weight 和其他一些参数，如是否使用 GPU（cuda=args.cuda）以及损失类型
        self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda, reduction='mean').build_loss(mode=args.loss_type)
        # 将构建的损失函数 self.criterion 与模型 self.model 和优化器 self.optimizer
        # 关联在一起。这样，在训练过程中，模型的输出与真实标签之间的差距将通过这个损失函数来计算，并且优化器将用来更新模型的权重以减小这个差距，从而实现模型的训练。
        self.model, self.optimizer = model, optimizer
        
        # Define Evaluator（定义模型的评估器）
        self.evaluator = Evaluator(self.nclass,self.args)
        # Define lr scheduler （定义学习率调度器）
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                            args.epochs, len(self.train_loader))

        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()

        # Resuming checkpoint
        # 恢复模型训练时的检查点，就是如果模型训练出现中断等情况，可以调整参数，定义检查点的路径
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

        # Clear start epoch if fine-tuning
        # 是否进行了微调（fine-tuning）。如果进行了微调，它将设置 args.start_epoch 的值为0
        if args.ft:
            args.start_epoch = 0

    def training(self, epoch):
        train_loss = 0.0#初始化训练损失为0
        self.model.train()#将模型设置为训练模式
        tbar = tqdm(self.train_loader)#设置进度条
        num_img_tr = len(self.train_loader)#获取训练数据集中的图像数量，用于计算平均损失。
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']   #从数据批次中获取输入图像和对应的标签。
            # print("输出拼接的图像尺寸",image.shape)
            # print(target)
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            # 用于跟踪模型在验证集上的最佳性能指标（通常是验证集上的平均IoU，也称为mIoU）。
            # 它的作用是在每个训练周期结束后，比较当前模型在验证集上的性能与历史最佳性能，
            # 如果当前性能优于历史最佳性能，则更新self.best_pred为当前性能。
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()#将优化器的梯度缓冲区清零
            output= self.model(image)#通过模型前向传播计算输出。
            # print(output.shape)
            loss = self.criterion(output, target)#计算模型输出与目标标签之间的损失
            loss.backward()#反向传播，计算梯度。
            self.optimizer.step()#优化器更新模型参数。
            train_loss += loss.item()#累积当前批次的损失值到train_loss。
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))#更新进度条的描述，显示当前的平均训练损失。
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)#将当前批次的损失值记录到Tensorboard中，以便可视化。
        #到此一个训练批次结束，进行下一个训练批次-->再到整个有一个epoch结束
        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)#计算并记录整个epoch的训练损失：
        self.saver.save_loss(train_loss)#保存训练损失到文件。
        self.saver.save2Png("loss")#保存训练损失的图表到文件。
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)
        is_best = False
        # 如果当前epoch的性能比之前的最佳性能好，保存模型的检查点。
        # 这个检查点包括模型的权重、优化器状态和最佳性能值。如果is_best为True，
        # 表示当前epoch的性能最佳，会额外保存一个带有"best"标签的检查点。这个检查点通常用于在训练过程中选择性地保存最佳模型。
        self.saver.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': self.model.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_pred': self.best_pred,
        }, is_best,'new_checkpoint.pth.tar')


    # 验证，评估
    def prediction(self):

        # Initialize lists to store precision and recall values
        precisions = []
        recalls = []

        self.model.eval()#将模型切换到评估模式。这通常会影响某些层（如Dropout和Batch Normalization层）的行为，以确保在推理时不会应用随机性。
        self.evaluator.reset()#重置一个用于评估模型性能的评估器对象。这个评估器可能用于计算像像素准确率、mIoU等指标。
        tbar = tqdm(self.test_loader, desc='\r')#循环遍历测试数据集（self.test_loader）
        test_loss = 0.0
        pure_inf_time = 0
        avarge_inf_time = 0
        
        # the first several iterations may be very slow so skip them
        num_warmup = 5
        # 汇总所有预测结果
        all_predictions = []
        all_target = []
        for i, sample in enumerate(tbar):

            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            # 当你在GPU上执行一系列的操作时，有时候需要确保某些操作已经完成，然后才能执行接下来的操作。
            # 这就是使用 torch.cuda.synchronize() 的时机。这个函数会强制等待GPU上的所有操作完成，然后才会继续执行后面的代码。
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            with torch.no_grad():#上下文管理器，禁用梯度计算，因为在推理阶段不需要计算梯度。
                output = self.model(image)#对输入图像进行前向传播，生成预测输出
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time
            # 计算推理时间（elapsed）以及平均推理时间（avarge_inf_time）
            if i >= num_warmup:
                pure_inf_time += elapsed
                avarge_inf_time = pure_inf_time/ (i+1 - num_warmup)
            #使用定义的损失函数（self.criterion）计算预测输出（output）与真实标签（target）之间的损失。
            loss = self.criterion(output, target)
            #累积
            test_loss += loss.item()

            # 对 output 进行 softmax 处理
            probabilities = F.softmax(output, dim=1)

            # 获取类别为 1 的概率图，形状为 b*256*256
            class_1_probabilities = probabilities[:, 1, :, :]

            # 将概率图添加到预测结果列表中
            all_predictions.append(class_1_probabilities)
            all_target.append(target)

            #将格式化后的测试损失信息更新到进度条的描述信息中，以便用户可以实时监视测试过程中的损失值。
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            #将预测输出（output）从GPU移动到CPU，并将其转换为NumPy数组。
            # output 是你的模型输出形状为 b*2*256*256
            # probabilities = F.softmax(output, dim=1)
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            # print(pred)

            #使用np.argmax获取每个像素点的最可能的类别索引，以得到最终的预测结果（pred）
            pred = np.argmax(pred, axis=1)
            # print(pred.shape)
            # 保存预测结果
            self.saver.save_images(pred, self.file_list, i)
            # Add batch sample into evaluator
            #使用真实标签（target）和预测结果（pred）更新评估器的性能指标。
            self.evaluator.add_batch(target, pred)



        # 像素准确率

        Acc = self.evaluator.Pixel_Accuracy()
        # 类别准确率
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        # 平均交并比
        mIoU = self.evaluator.Mean_Intersection_over_Union()

        IoU1 = self.evaluator.IoU1()
        IoU0 = self.evaluator.IoU0()
        # 频权交并比
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        Kappa = self.evaluator.Kappa()
        OA = self.evaluator.Overall_Accuracy()
        cmIoU = self.evaluator.Mean_Intersection_over_Union_Class()
        mF1,f1_scores = self.evaluator.calculate_f1_score()
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}, Kappa:{}, mF1:{}, F1_1:{} IoU:{}, OA:{} Inference_time: {}".format
              (Acc, Acc_class, mIoU, FWIoU, Kappa, mF1, f1_scores[1], IoU1, OA, avarge_inf_time))
        print("ever class miou:{} ".format(cmIoU))
        print("ever class miou:",IoU1,IoU0)
        print('Loss: %.3f' % test_loss)



    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output= self.model(image)
            loss = self.criterion(output, target)
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            self.evaluator.add_batch(target, pred)


        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        Kappa = self.evaluator.Kappa()
        mF1,f1_scores = self.evaluator.calculate_f1_score()
        IoU1 = self.evaluator.IoU1()
        #使用 self.writer 将这些性能指标记录到日志中。通过可视化工具Tensorboard
        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
        self.writer.add_scalar('val/Kappa', Kappa, epoch)
        self.writer.add_scalar('val/mF1', mF1, epoch)
        # 保存 mIoU 和检查点
        self.saver.save_miou(mIoU)
        self.saver.save2Png("miou")
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}, Kappa:{}, mF1:{},F1_1:{},IoU1:{}".format(Acc, Acc_class, mIoU, FWIoU, Kappa, mF1,f1_scores[1], IoU1))
        print('Loss: %.3f' % test_loss)

        #如果当前的 mIoU 值超过了历史最佳值（self.best_pred），则将其视为新的最佳值，并保存模型检查点。
        new_pred = mIoU
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)

def main():
    parser = argparse.ArgumentParser(description="PyTorch UNet Training")

    # 数据集的类别数目，用于指定模型的输出类别数
    parser.add_argument('--num-class', type=int, default=16,
                        help='dataset num classes')
    # 模型的骨干网络架构
    parser.add_argument('--backbone', type=str, default='resnet50',
                        choices=['resnet50','resnet101','resnet152', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    # 选择要训练的分割模型，可以从给定的架构中选择一个进行训练
    parser.add_argument('--train_model', type=str, default='MFEPNet',
                        choices=['ResNet50','MSFRM','CTCTNet','TEEM','CCAFM','MFEPNet'],
                        help = 'select a model for training. ')
    # 网络的输出步幅，用于控制特征图的分辨率
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    # 选择要使用的数据集
    parser.add_argument('--dataset', type=str, default='pascalHelan',
                        choices=['pascalHelan','pascalRregular','pascalIrregular'],
                        help='dataset name (default: pascal)')
    #参数用于指定数据集的存储路径。
    parser.add_argument('--data_path', type=str, default='/data')
    # 是否使用 SBD（Semantic Boundaries Dataset）数据集，用于训练
    parser.add_argument('--use-sbd', action='store_true', default=False,
                        help='whether to use SBD dataset (default: True)')
    # 是否使用注意力机制。
    parser.add_argument('--use-attens', type=int, default=1, choices=[0, 1],
                        help='whether to use attens (default: True)')
    # 数据加载器线程数。
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    # 输入图像的通道数 
    parser.add_argument('--in-channels', type=int, default=3,
                        metavar='N', help='dataloader threads')
    # 基本图像大小
    parser.add_argument('--base-size', type=int, default=256,
                        help='base image size')
    # 裁剪图像大小
    parser.add_argument('--crop-size', type=int, default=256,
                        help='crop image size')
    # 是否使用批量归一化，主要是针对多个CPU上的参数一致性的问题
    # 在 BatchNorm 中，每个训练小批量（batch）的输入数据都会被标准化，使其具有零均值和单位方差。
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    # 是否冻结 Batch Normalization 参数。
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    
    # 指定是进行训练还是测试
    parser.add_argument('--train-test', type=str, default='False',
                        choices=['False','True'],
                        help='train or pred')
    # 选择多尺度类型。
    # 是指在输入数据的不同尺度下运行模型以提高性能或鲁棒性的一种技术
    # 可以提高模型对不同尺度物体的泛化能力
    # 存储不同尺寸的图片
    parser.add_argument('--suffix', type=list, default=['0','1','2'],
                        help='choose muti scale type')
    # 损失函数类型，可以是交叉熵 ('ce') 或 Focal 损失 ('focal')
    parser.add_argument('--loss-type', type=str, default='bce',
                        choices=['ce', 'focal', 'bce'],
                        help='loss func type (default: ce)')
    # training hyper params
    # 训练的总轮数。
    parser.add_argument('--epochs', type=int, default=500, metavar='N',
                        help='number of epochs to train (default: auto)')
    # 训练的起始轮数
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    # 训练时的批量大小
    parser.add_argument('--batch-size', type=int, default=4,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    # 测试时的批量大小
    parser.add_argument('--test-batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    # 是否使用平衡权重
    parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')
    # 是否使用边缘检测
    parser.add_argument('--edge-detection', type=bool, default=False,
                        help='whether to use edge detection')
    # optimizer params
    # 优化器类型
    parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam', 'AdamW', 'Adan'],
                        help='optimizer (default: SGD)')
    # 学习率
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (default: auto)')
    # 学习率调度器
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    # 优化器动量
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    # parser.add_argument('--weight-decay', type=float, default=5e-4,
    #                     metavar='M', help='w-decay (default: 5e-4)')

    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    # 随机种子。
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # 用于梯度剪裁的阈值。
    parser.add_argument('--max-grad-norm', type=float, default=0.0,
                        help='if the l2 norm is large than this hyper-parameter, then we clip the gradient  (default: 0.0, no gradient clip)')
    # 权重衰减（weight decay）参数。
    parser.add_argument('--weight-decay', type=float, default=0.02, metavar='M',
                        help='weight decay, similar one used in AdamW (default: 0.02)')
    # 优化器的 epsilon 参数。
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='optimizer epsilon to avoid the bad case where second-order moment is zero (default: None, use opt default 1e-8 in adan)')
    # 优化器的 beta 参数。
    parser.add_argument('--opt-betas', default=[0.98, 0.92, 0.99], type=float, nargs='+', metavar='BETA',
                        help='optimizer betas in Adan (default: None, use opt default [0.98, 0.92, 0.99] in Adan)')
    parser.add_argument('--no-prox', action='store_true', default=False,
                        help='whether perform weight decay like AdamW (default=False)')
    # checking point
    # 恢复模型训练的检查点文件路径。
    parser.add_argument('--resume', type=str, default=None, 
                        help='put the path to resuming file if needed')
    # finetuning pre-trained models
    # 是否进行在不同数据集上的微调。
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    # evaluation option
    # 模型评估的间隔。
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    # 是否跳过训练过程中的验证。
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epoches = {
            'coco': 30,
            'cityscapes': 200,
            'pascal': 100,
            'ms_pascal': 300,
            'pascalpred':50,
        }
        args.epochs = epoches[args.dataset.lower()]

    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    if args.lr is None:
        lrs = {
            'coco': 0.1,
            'cityscapes': 0.01,
            'pascal': 0.003,
            'ms_pascal': 0.003,
            'pascalpred':0.007,
        }
#         args.lr = lrs[args.dataset.lower()] / (4 * len(args.gpu_ids)) * args.batch_size
        args.lr = 0.003

    print(args)
    # 设置PyTorch的随机数生成种子（random seed）。
    # 随机数生成在深度学习中经常被用来初始化模型参数、数据增强、数据划分等许多地方。通过设置随机数种子，
    # 可以使实验具有可重复性，即每次运行相同的代码时，生成的随机数序列都是相同的。
    # 就是我们要对比不同的神经网络模型，以此来比较神经网络模型的优略性
    torch.manual_seed(args.seed)
    trainer = Trainer(args)

    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)

    if args.train_test == "True":
        for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
            trainer.training(epoch)
            if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
                trainer.validation(epoch)
    else:
        trainer.prediction()

    trainer.writer.close()

if __name__ == "__main__":
   main()

