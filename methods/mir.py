import logging
import random
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from methods.er_baseline import ER
from utils.data_loader import cutmix_data, ImageDataset, StreamDataset

logger = logging.getLogger()
writer = SummaryWriter("tensorboard")

class MIR(ER):
    def __init__(
        self, criterion, device, train_transform, test_transform, n_classes, **kwargs
    ):
        super().__init__(
            criterion, device, train_transform, test_transform, n_classes, **kwargs
        )
        self.cand_size = kwargs['mir_cands']

    def online_train(self, sample, batch_size, n_worker, iterations=1, stream_batch_size=1):
        self.model.train()
        total_loss, correct, num_data = 0.0, 0.0, 0.0
        assert stream_batch_size > 0
        sample_dataset = StreamDataset(sample, dataset=self.dataset, transform=self.train_transform,
                                       cls_list=self.exposed_classes, data_dir=self.data_dir, device=self.device,
                                       transform_on_gpu=self.gpu_transform)

        for i in range(iterations):
            stream_data = sample_dataset.get_data()
            str_x = stream_data['image']
            str_y = stream_data['label']
            x = str_x.to(self.device)
            y = str_y.to(self.device)
            logit = self.model(x)
            loss = self.criterion(logit, y)
            self.optimizer.zero_grad()
            loss.backward()
            grads = {}
            for name, param in self.model.named_parameters():
                grads[name] = param.grad.data

            if len(self.memory) > 0:
                memory_batch_size = min(len(self.memory), batch_size - stream_batch_size)

                lr = self.optimizer.param_groups[0]['lr']
                new_model = copy.deepcopy(self.model)
                for name, param in new_model.named_parameters():
                    param.data = param.data - lr * grads[name]

                memory_cands, memory_cands_test = self.memory.get_two_batches(min(self.cand_size, len(self.memory)), test_transform=self.test_transform)
                x = memory_cands_test['image']
                y = memory_cands_test['label']
                x = x.to(self.device)
                y = y.to(self.device)
                with torch.no_grad():
                    if self.use_amp:
                        with torch.cuda.amp.autocast():
                            logit_pre = self.model(x)
                            logit_post = new_model(x)
                            pre_loss = F.cross_entropy(logit_pre, y, reduction='none')
                            post_loss = F.cross_entropy(logit_post, y, reduction='none')
                            scores = post_loss - pre_loss
                    else:
                        logit_pre = self.model(x)
                        logit_post = new_model(x)
                        pre_loss = F.cross_entropy(logit_pre, y, reduction='none')
                        post_loss = F.cross_entropy(logit_post, y, reduction='none')
                        scores = post_loss - pre_loss
                selected_samples = torch.argsort(scores, descending=True)[:memory_batch_size]
                mem_x = memory_cands['image'][selected_samples]
                mem_y = memory_cands['label'][selected_samples]
                x = torch.cat([str_x, mem_x])
                y = torch.cat([str_y, mem_y])
                x = x.to(self.device)
                y = y.to(self.device)

            self.optimizer.zero_grad()
            logit, loss = self.model_forward(x, y)
            _, preds = logit.topk(self.topk, 1, True, True)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            self.update_schedule()

            total_loss += loss.item()
            correct += torch.sum(preds == y.unsqueeze(1)).item()
            num_data += y.size(0)

        return total_loss / iterations, correct / num_data