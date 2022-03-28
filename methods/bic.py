# @inproceedings{wu2019large,
#   title={Large scale incremental learning},
#   author={Wu, Yue and Chen, Yinpeng and Wang, Lijuan and Ye, Yuancheng and Liu, Zicheng and Guo, Yandong and Fu, Yun},
#   booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
#   pages={374--382},
#   year={2019}
# }
import logging
import copy
from copy import deepcopy

import pandas as pd
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader

from methods.er_baseline import ER
from utils.data_loader import ImageDataset, cutmix_data, StreamDataset
from utils.train_utils import select_model, cycle
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger()
writer = SummaryWriter("tensorboard")


class BiasCorrectionLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1, bias=True)
        self.linear.weight.data.fill_(1.0)
        self.linear.bias.data.fill_(0.0)

    def forward(self, x):
        correction = self.linear(x.unsqueeze(dim=2))
        correction = correction.squeeze(dim=2)
        return correction


class BiasCorrection(ER):
    def __init__(
        self, criterion, device, train_transform, test_transform, n_classes, **kwargs
    ):
        """
        self.valid_list: valid set which is used for training bias correction layer.
        self.memory_list: training set only including old classes. As already mentioned in the paper,
            memory list and valid list are exclusive.
        self.bias_layer_list - the list of bias correction layers. The index of the list means the task number.
        """
        super().__init__(
            criterion, device, train_transform, test_transform, n_classes, **kwargs
        )
        self.prev_model = select_model(
            self.model_name, self.dataset, 1
        )
        self.bias_layer = None
        self.valid_list = []

        self.valid_size = round(self.memory_size * 0.1)
        self.memory_size = self.memory_size - self.valid_size

        self.n_tasks = kwargs["n_tasks"]
        self.bias_layer_list = []
        for _ in range(self.n_tasks):
            bias_layer = BiasCorrectionLayer().to(self.device)
            self.bias_layer_list.append(bias_layer)
        self.distilling = kwargs["distilling"]

        self.val_per_cls = self.valid_size
        self.val_full = False
        self.cur_iter = 0
        self.bias_labels = []

    def online_before_task(self, cur_iter):
        super().online_before_task(cur_iter)
        self.cur_iter = cur_iter
        self.bias_labels.append([])

    def online_after_task(self, cur_iter):
        if self.distilling:
            self.prev_model = deepcopy(self.model)

    def online_step(self, sample, sample_num, n_worker):
        if sample['klass'] not in self.exposed_classes:
            self.add_new_class(sample['klass'])

        use_sample = self.online_valid_update(sample)
        self.num_updates += self.online_iter

        if use_sample:
            self.temp_batch.append(sample)
            if len(self.temp_batch) == self.temp_batchsize:
                train_loss, train_acc = self.online_train(self.temp_batch, self.batch_size, n_worker,
                                                          iterations=int(self.num_updates),
                                                          stream_batch_size=self.temp_batchsize)
                self.report_training(sample_num, train_loss, train_acc)
                for stored_sample in self.temp_batch:
                    self.update_memory(stored_sample)
                self.temp_batch = []
                self.num_updates -= int(self.num_updates)

    def add_new_class(self, class_name):
        if self.distilling:
            self.prev_model = deepcopy(self.model)
        self.exposed_classes.append(class_name)
        self.num_learned_class = len(self.exposed_classes)
        prev_weight = copy.deepcopy(self.model.fc.weight.data)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_learned_class).to(self.device)

        with torch.no_grad():
            if self.num_learned_class > 1:
                self.model.fc.weight[:self.num_learned_class - 1] = prev_weight
        for param in self.optimizer.param_groups[1]['params']:
            if param in self.optimizer.state.keys():
                del self.optimizer.state[param]
        del self.optimizer.param_groups[1]
        self.optimizer.add_param_group({'params': self.model.fc.parameters()})
        self.memory.add_new_class(cls_list=self.exposed_classes)

        self.bias_labels[self.cur_iter].append(self.num_learned_class - 1)
        if self.num_learned_class > 1:
            self.online_reduce_valid(self.num_learned_class)

        if 'reset' in self.sched_name:
            self.update_schedule(reset=True)

    def online_reduce_valid(self, num_learned_class):
        self.val_per_cls = self.valid_size//num_learned_class
        val_df = pd.DataFrame(self.valid_list)
        valid_list = []
        for klass in val_df["klass"].unique():
            class_val = val_df[val_df.klass == klass]
            if len(class_val) > self.val_per_cls:
                new_class_val = class_val.sample(n=self.val_per_cls)
            else:
                new_class_val = class_val
            valid_list += new_class_val.to_dict(orient="records")
        self.valid_list = valid_list
        self.val_full = False

    def online_valid_update(self, sample):
        val_df = pd.DataFrame(self.valid_list, columns=['klass', 'file_name', 'label'])
        if not self.val_full:
            if len(val_df[val_df["klass"] == sample["klass"]]) < self.val_per_cls:
                self.valid_list.append(sample)
                if len(self.valid_list) == self.val_per_cls*self.num_learned_class:
                    self.val_full = True
                use_sample = False
            else:
                use_sample = True
        else:
            use_sample = True
        return use_sample

    def online_train(self, sample, batch_size, n_worker, iterations=1, stream_batch_size=1):
        self.model.train()
        total_loss, distill_loss, classify_loss, correct, num_data = 0.0, 0.0, 0.0, 0.0, 0.0

        if stream_batch_size > 0:
            sample_dataset = StreamDataset(
                sample,
                dataset=self.dataset,
                transform=self.train_transform,
                cls_list=self.exposed_classes,
                data_dir=self.data_dir,
                device=self.device,
                transform_on_gpu=self.gpu_transform
            )
        if len(self.memory) > 0 and batch_size - stream_batch_size > 0:
            memory_batch_size = min(len(self.memory), batch_size - stream_batch_size)

        for i in range(iterations):
            x = []
            y = []
            if stream_batch_size > 0:
                stream_data = sample_dataset.get_data()
                x.append(stream_data['image'])
                y.append(stream_data['label'])
            if len(self.memory) > 0 and batch_size - stream_batch_size > 0:
                memory_data = self.memory.get_batch(memory_batch_size)
                x.append(memory_data['image'])
                y.append(memory_data['label'])
            x = torch.cat(x)
            y = torch.cat(y)

            x = x.to(self.device)
            y = y.to(self.device)
            do_cutmix = self.cutmix and np.random.rand(1) < 0.5
            if do_cutmix:
                x, labels_a, labels_b, lam = cutmix_data(x=x, y=y, alpha=1.0)
                if self.cur_iter != 0:
                    if self.distilling:
                        with torch.no_grad():
                            if self.use_amp:
                                with torch.cuda.amp.autocast():
                                    logit_old = self.prev_model(x)
                            else:
                                logit_old = self.prev_model(x)
                                logit_old = self.online_bias_forward(logit_old, self.cur_iter - 1)

                self.optimizer.zero_grad()
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        logit_new = self.model(x)
                        loss_c = lam * self.criterion(logit_new, labels_a) + (1 - lam) * self.criterion(
                        logit_new, labels_b)
                else:
                    logit_new = self.model(x)
                    loss_c = lam * self.criterion(logit_new, labels_a) + (1 - lam) * self.criterion(
                        logit_new, labels_b)

            else:
                if self.cur_iter != 0:
                    if self.distilling:
                        with torch.no_grad():
                            if self.use_amp:
                                with torch.cuda.amp.autocast():
                                    logit_old = self.prev_model(x)
                                    logit_old = self.online_bias_forward(logit_old, self.cur_iter - 1)
                            else:
                                logit_old = self.prev_model(x)
                                logit_old = self.online_bias_forward(logit_old, self.cur_iter - 1)
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        logit_new = self.model(x)
                        loss_c = self.criterion(logit_new, y)
                else:
                    logit_new = self.model(x)
                    loss_c = self.criterion(logit_new, y)

            if self.distilling:
                if self.cur_iter == 0:
                    loss_d = torch.tensor(0.0).to(self.device)
                else:
                    loss_d = self.distillation_loss(logit_old, logit_new[:, : logit_old.size(1)])
            else:
                loss_d = torch.tensor(0.0).to(self.device)

            _, preds = logit_new.topk(self.topk, 1, True, True)
            loss = loss_c + loss_d
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                self.optimizer.step()
            self.update_schedule()
            total_loss += loss.item()
            distill_loss += loss_d.item()
            classify_loss += loss_c.item()
            correct += torch.sum(preds == y.unsqueeze(1)).item()
            num_data += y.size(0)

        return total_loss / iterations, correct / num_data

    def online_bias_forward(self, input, iter):
        bias_labels = self.bias_labels[iter]
        bias_layer = self.bias_layer_list[iter]
        if len(bias_labels) > 0:
            input[:, bias_labels] = bias_layer(input[:, bias_labels])
        return input

    def online_evaluate(self, test_list, sample_num, batch_size, n_worker):
        self.online_bias_correction()
        test_df = pd.DataFrame(test_list)
        exp_test_df = test_df[test_df['klass'].isin(self.exposed_classes)]
        test_dataset = ImageDataset(
            exp_test_df,
            dataset=self.dataset,
            transform=self.test_transform,
            cls_list=self.exposed_classes,
            data_dir=self.data_dir
        )
        test_loader = DataLoader(
            test_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=n_worker,
        )
        total_correct, total_num_data, total_loss = (
            0.0,
            0.0,
            0.0,
        )
        correct_l = torch.zeros(self.n_classes)
        num_data_l = torch.zeros(self.n_classes)
        label = []

        self.model.eval()
        self.bias_layer.eval()

        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x = data["image"]
                xlabel = data["label"]
                x = x.to(self.device)
                logit = self.model(x)
                logit = self.online_bias_forward(logit, self.cur_iter)
                logit = logit.detach().cpu()
                loss = self.criterion(logit, xlabel)
                pred = torch.argmax(logit, dim=-1)
                _, preds = logit.topk(self.topk, 1, True, True)
                total_correct += torch.sum(preds == xlabel.unsqueeze(1)).item()
                total_num_data += xlabel.size(0)

                xlabel_cnt, correct_xlabel_cnt = self._interpret_pred(xlabel, pred)
                correct_l += correct_xlabel_cnt.detach().cpu()
                num_data_l += xlabel_cnt.detach().cpu()

                total_loss += loss.item()
                label += xlabel.tolist()

        avg_acc = total_correct / total_num_data
        avg_loss = total_loss / len(test_loader)

        cls_acc = (correct_l / (num_data_l + 1e-5)).numpy().tolist()
        eval_dict = {"avg_loss": avg_loss, "avg_acc": avg_acc, "cls_acc": cls_acc}

        writer.add_scalar(f"test/loss", eval_dict["avg_loss"], sample_num)
        writer.add_scalar(f"test/acc", eval_dict["avg_acc"], sample_num)
        logger.info(
            f"Test | Sample # {sample_num} | test_loss {eval_dict['avg_loss']:.4f} | test_acc {eval_dict['avg_acc']:.4f} | "
        )
        return eval_dict

    def online_bias_correction(self, n_iter=256, batch_size=100, n_worker=4):
        self.bias_layer_list[self.cur_iter] = BiasCorrectionLayer().to(self.device)
        self.bias_layer = self.bias_layer_list[self.cur_iter]

        if self.val_full and self.cur_iter > 0 and len(self.bias_labels[self.cur_iter]) > 0:
            val_df = pd.DataFrame(self.valid_list)
            val_dataset = ImageDataset(val_df, dataset=self.dataset, transform=self.test_transform,
                                       cls_list=self.exposed_classes, data_dir=self.data_dir, preload=True)
            bias_loader = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=n_worker)
            criterion = self.criterion
            self.bias_layer = self.bias_layer_list[self.cur_iter]
            optimizer = torch.optim.Adam(params=self.bias_layer.parameters(), lr=0.001)
            self.model.eval()
            total_loss = None
            model_out = []
            xlabels = []
            for i, data in enumerate(bias_loader):
                x = data["image"]
                xlabel = data["label"]
                x = x.to(self.device)
                xlabel = xlabel.to(self.device)
                with torch.no_grad():
                    out = self.model(x)
                model_out.append(out.detach().cpu())
                xlabels.append(xlabel.detach().cpu())
            for iteration in range(n_iter):
                self.bias_layer.train()
                total_loss = 0.0
                for i, out in enumerate(model_out):
                    logit = self.online_bias_forward(out.to(self.device), self.cur_iter)
                    xlabel = xlabels[i]
                    loss = criterion(logit, xlabel.to(self.device))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                logger.info(
                    "[Stage 2] [{}/{}]\tloss: {:.4f}\talpha: {:.4f}\tbeta: {:.4f}".format(
                        iteration + 1,
                        n_iter,
                        total_loss,
                        self.bias_layer.linear.weight.item(),
                        self.bias_layer.linear.bias.item(),
                    )
                )
            assert total_loss is not None
            self.print_bias_layer_parameters()

    def distillation_loss(self, old_logit, new_logit):
        # new_logit should have same dimension with old_logit.(dimension = n)
        assert new_logit.size(1) == old_logit.size(1)
        T = 2
        old_softmax = torch.softmax(old_logit / T, dim=1)
        new_log_softmax = torch.log_softmax(new_logit / T, dim=1)
        loss = -(old_softmax * new_log_softmax).sum(dim=1).mean()
        return loss

    def print_bias_layer_parameters(self):
        for i, layer in enumerate(self.bias_layer_list):
            logger.info(
                "[{}] alpha: {:.4f}, beta: {:.4f}".format(
                    i, layer.linear.weight.item(), layer.linear.bias.item()
                )
            )
