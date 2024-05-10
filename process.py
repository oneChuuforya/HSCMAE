import time
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

from datautils import load_UCR
from loss import CE, Align, Reconstruct
from torch.optim.lr_scheduler import LambdaLR
import torch.nn as nn
from classification import fit_lr, get_rep_with_label
from timm import create_model
from typing import List
from sklearn.ensemble import IsolationForest

class Trainer():
    def __init__(self, args, model, train_loader, train_linear_loader, test_loader,target_loader, verbose=False):
        self.args = args
        self.verbose = verbose
        self.device = args.device
        self.print_process(self.device)
        self.model = model.to(torch.device(self.device))
        # self.model = model.cuda()
        print('model cuda')
        self.target_loader = target_loader
        self.train_loader = train_loader
        self.train_linear_loader = train_linear_loader
        self.test_loader = test_loader
        self.lr_decay = args.lr_decay_rate
        self.lr_decay_steps = args.lr_decay_steps

        self.alpha = args.alpha
        self.beta = args.beta

        self.test_cr = torch.nn.CrossEntropyLoss()
        self.num_epoch = args.num_epoch
        self.num_epoch_pretrain = args.num_epoch_pretrain
        self.eval_per_steps = args.eval_per_steps
        self.save_path = args.save_path
        if self.num_epoch:
            self.result_file = open(self.save_path + '/result_fineALL.txt', 'w')
            self.result_file.close()
        self.best_results = []
        self.step = 0
        self.best_metric = -1e9
        self.metric = 'acc'

    def pretrain(self):
        print('pretraining')
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr)
        eval_acc = 0

        best_loss = 10000
        for epoch in range(self.num_epoch_pretrain):
            self.model.train()
            tqdm_dataloader = tqdm(self.train_loader)
            loss_sum = 0
            loss_mse = 0
            loss_ce = 0
            hits_sum = 0
            NDCG_sum = 0
            for idx, batch in enumerate(tqdm_dataloader):
                batch = [x.to(self.device) for x in batch]
                self.optimizer.zero_grad()
                loss,global_loss = self.model.pretrain_forward(batch[0])
                loss += global_loss

                loss.backward()
                self.optimizer.step()

                loss_sum += loss.item()
            print('pretrain epoch{0}, loss{1}, mse{2}, ce{3}, hits{4}, ndcg{5}'.format(epoch + 1, loss_sum / (idx + 1),
                                                                                       loss_mse / (idx + 1),
                                                                                       loss_ce / (idx + 1), hits_sum,
                                                                                       NDCG_sum / (idx + 1)))
            result_file = open(self.save_path + '/pretrain_result.txt', 'a+')
            print('pretrain epoch{0}, loss{1}, mse{2}, ce{3}, hits{4}, ndcg{5}'.format(epoch + 1, loss_sum / (idx + 1),
                                                                                       loss_mse / (idx + 1),
                                                                                       loss_ce / (idx + 1), hits_sum,
                                                                                       NDCG_sum / (idx + 1)),
                  file=result_file)
            result_file.close()

            if (epoch + 1) % 1 == 0:
                train_rep, train_label = get_rep_with_label(self.model, self.train_linear_loader)
                test_rep, test_label = get_rep_with_label(self.model, self.test_loader)
                clf = fit_lr(train_rep, train_label)
                acc = clf.score(test_rep, test_label)
                print(acc)
                result_file = open(self.save_path + '/linear_result_fineALL.txt', 'a+')
                print('epoch{0}, acc{1}'.format(epoch, acc), file=result_file)
                result_file.close()
                if acc > eval_acc:
                    eval_acc = acc
                    torch.save(self.model.sparse_encoder.sp_cnn.state_dict(), self.save_path + '/pretrain_model.pkl')

    def finetune(self):
        print('finetune')
        denseEnc = create_model('convnext_tiny1',unet_layers=self.args.unet_layers,
                                depths=self.args.unet_depths, dims=self.args.unet_dims,bigkernel = self.args.bigker,
                                num_classes=self.args.num_class, drop_path_rate=0.1).to('cuda')
        # print(f'[load_checkpoint] ep_start={ep_start}, performance_desc={performance_desc}')
        if self.args.load_pretrained_model:
            print('load pretrained model')
            checkpoint = torch.load(self.save_path + '/pretrain_model.pkl', map_location=self.device)
            missing, unexpected = denseEnc.load_state_dict(checkpoint.get('module', checkpoint), strict=False)
            print(f'[load_checkpoint] missing_keys={missing}')
            print(f'[load_checkpoint] unexpected_keys={unexpected}')
        denseEnc.eval()


        self.model.linear_proba = False
        if self.args.freeze:
            freeze_layers = ("downsample_layers", "stages")
            for name, param in denseEnc.named_parameters():

                if name.split(".")[0] in freeze_layers:
                    param.requires_grad = False
        self.optimizer = torch.optim.Adam(denseEnc.parameters(), lr=self.args.lr)
        if self.args.adjust:
            adl1 = nn.Linear(self.args.target_shape[0]*self.args.target_shape[1],128*9).to('cuda')
            denseEnc1 = nn.Sequential(adl1, denseEnc)
            self.optimizer = torch.optim.Adam(denseEnc1.parameters(), lr=self.args.lr)

        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda step: self.lr_decay ** step, verbose=self.verbose)
        for epoch in range(self.num_epoch):
            if self.args.adjust:
                loss_epoch, time_cost = self._train_one_epoch(denseEnc1)
            else:
                loss_epoch, time_cost = self._train_one_epoch(denseEnc)
            self.result_file = open(self.save_path + '/result_fineALL.txt', 'a+')
            self.print_process(
                'Finetune epoch:{0},loss:{1},training_time:{2}'.format(epoch + 1, loss_epoch, time_cost))
            print('Finetune train epoch:{0},loss:{1},training_time:{2}'.format(epoch + 1, loss_epoch, time_cost),
                  file=self.result_file)
            self.result_file.close()

        self.print_process(self.best_results)
        return self.best_metric


    def _train_one_epoch(self,denseEnc):
        t0 = time.perf_counter()
        denseEnc.train()
        loss_sum = 0
        if self.args.adjust:
            tqdm_dataloader = tqdm(self.target_loader)
        else:
            tqdm_dataloader = tqdm(self.train_linear_loader) if self.verbose else self.train_linear_loader
        for idx, batch in enumerate(tqdm_dataloader):
            batch = [x.to(self.device) for x in batch]
            self.cr = CE(denseEnc,self.args.adjust)
            self.optimizer.zero_grad()
            loss = self.cr.compute(batch)
            loss_sum += loss.item()
            loss.backward()

            self.optimizer.step()

            self.step += 1
        # if self.step % self.eval_per_steps == 0:
        metric = self.eval_model(denseEnc)
        # self.print_process(metric)
        self.result_file = open(self.save_path + '/result_fineALL.txt', 'a+')
        print('step{0}'.format(self.step), file=self.result_file)
        print(metric, file=self.result_file)

        if metric[self.metric] >= self.best_metric:
            torch.save(denseEnc.state_dict(), self.save_path + '/model.pkl')
            # result_file = open(self.save_path + '/best_result_Ulayer'+str(self.args.unet_layers)+'fr_'+str(self.args.freeze)+'.txt', 'a+')
            result_file = open(self.save_path + '/best_result_MK' + str(self.args.mask_ratio) + 'fr_' + str(
                self.args.freeze) + '.txt', 'a+')
            print('saving model of step{0}'.format(self.step), file=self.result_file)
            print('saving model of step{0}'.format(self.step), file=result_file)
            self.best_results = metric
            print(metric, file=result_file)
            # print(metric,'<-BEST REPORT')
            result_file.close()
            self.best_metric = metric[self.metric]
            torch.save(denseEnc.state_dict(), self.save_path + '/finetune_model.pkl')
        self.result_file.close()
        denseEnc.train()

        return loss_sum / (idx + 1), time.perf_counter() - t0


    def eval_model(self,denseEnc):
        denseEnc.eval()

        tqdm_data_loader = tqdm(self.test_loader) if self.verbose else self.test_loader
        metrics = {'acc': 0, 'f1': 0}
        pred = []
        label = []
        test_loss = 0

        with torch.no_grad():
            for idx, batch in enumerate(tqdm_data_loader):
                batch = [x.to(self.device) for x in batch]
                ret = self.compute_metrics(batch,denseEnc)
                if len(ret) == 2:
                    pred_b, label_b = ret
                    pred += pred_b
                    label += label_b
                else:
                    pred_b, label_b, test_loss_b = ret
                    pred += pred_b
                    label += label_b
                    test_loss += test_loss_b.cpu().item()
        confusion_mat = self._confusion_mat(label, pred)
        # self.print_process(confusion_mat)
        self.result_file = open(self.save_path + '/result_fineALL.txt', 'a+')
        print(confusion_mat, file=self.result_file)
        self.result_file.close()
        if self.args.num_class == 2:
            metrics['f1'] = f1_score(y_true=label, y_pred=pred)
            metrics['precision'] = precision_score(y_true=label, y_pred=pred)
            metrics['recall'] = recall_score(y_true=label, y_pred=pred)
        else:
            metrics['f1'] = f1_score(y_true=label, y_pred=pred, average='macro')
            metrics['micro_f1'] = f1_score(y_true=label, y_pred=pred, average='micro')
        metrics['acc'] = accuracy_score(y_true=label, y_pred=pred)
        metrics['test_loss'] = test_loss / (idx + 1)
        return metrics

    def compute_metrics(self, batch,denseEnc):
        if len(batch) == 2:
            seqs, label = batch
            if self.args.adjust:
                seqs = seqs.reshape(label.size()[0], -1)
                out1 = denseEnc[0](seqs)
                out1 = out1.reshape(label.size()[0], 128, -1)
                scores = denseEnc[1](out1)
            else:
                scores = denseEnc(seqs)
        else:
            seqs1, seqs2, label = batch
            scores = denseEnc((seqs1, seqs2))
        _, pred = torch.topk(scores, 1)
        test_loss = self.test_cr(scores, label.view(-1).long())
        pred = pred.view(-1).tolist()
        return pred, label.tolist(), test_loss

    def _confusion_mat(self, label, pred):
        mat = np.zeros((self.args.num_class, self.args.num_class))
        for _label, _pred in zip(label, pred):
            mat[_label, _pred] += 1
        return mat

    def print_process(self, *x):
        if self.verbose:
            print(*x)

from pprint import pformat
def get_param_groups(model, nowd_keys=(), lr_scale=0.0):
    using_lr_scale = hasattr(model, 'get_layer_id_and_scale_exp') and 0.0 < lr_scale < 1.0
    print(f'[get_ft_param_groups][lr decay] using_lr_scale={using_lr_scale}, ft_lr_scale={lr_scale}')
    para_groups, para_groups_dbg = {}, {}

    for name, para in model.named_parameters():
        if not para.requires_grad:
            continue  # frozen weights
        if len(para.shape) == 1 or name.endswith('.bias') or any(k in name for k in nowd_keys):
            wd_scale, group_name = 0., 'no_decay'
        else:
            wd_scale, group_name = 1., 'decay'

        if using_lr_scale:
            layer_id, scale_exp = model.get_layer_id_and_scale_exp(name)
            group_name = f'layer{layer_id}_' + group_name
            this_lr_scale = lr_scale ** scale_exp
            dbg = f'[layer {layer_id}][sc = {lr_scale} ** {scale_exp}]'
        else:
            this_lr_scale = 1
            dbg = f'[no scale]'

        if group_name not in para_groups:
            para_groups[group_name] = {'params': [], 'weight_decay_scale': wd_scale, 'lr_scale': this_lr_scale}
            para_groups_dbg[group_name] = {'params': [], 'weight_decay_scale': wd_scale, 'lr_scale': dbg}
        para_groups[group_name]['params'].append(para)
        para_groups_dbg[group_name]['params'].append(name)

    for g in para_groups_dbg.values():
        g['params'] = pformat(', '.join(g['params']), width=200)

    print(f'[get_ft_param_groups] param groups = \n{pformat(para_groups_dbg, indent=2, width=250)}\n')
    return list(para_groups.values())
