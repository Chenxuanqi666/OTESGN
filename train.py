'''
Description: 
version: 
Author: chenxuanqi
Date: 2021-06-09 14:17:37
'''
import json
import os
import sys
import copy
import random
import logging
import argparse
import torch
import torch.nn as nn
import numpy as np
from sklearn import metrics
from time import strftime, localtime
from torch.utils.data import DataLoader
from transformers import BertModel, AdamW

from models.OTESGN import OTESGCN
from models.wo_SyntacticMask import SSEGCNBertClassifier_wo_SyntacticMask
from data_utils import SentenceDataset, build_tokenizer, build_embedding_matrix, Tokenizer4BertGCN, ABSAGCNData
from prepare_vocab import VocabHelp
import torch.nn.functional as F

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def read_data_split(opt):
    """
        读取数据并重新分配
    """
    data = []
    with open(opt.dataset_file['train'], 'r', encoding='utf-8') as f:
        data.extend(json.load(f))
    with open(opt.dataset_file['test'], 'r', encoding='utf-8') as f:
        data.extend(json.load(f))

    # 打乱数据
    random.shuffle(data)

    # 计算 9:1 分界点
    split_idx = int(len(data) * 0.9)
    train_data = data[:split_idx]
    test_data = data[split_idx:]

    print(f"新训练集: {len(train_data)} 条，新测试集: {len(test_data)} 条")

    # 保存到原始路径
    with open(opt.dataset_file['train'], 'w', encoding='utf-8') as f_train:
        f_train.write(json.dumps(train_data, indent=4))

    with open(opt.dataset_file['test'], 'w', encoding='utf-8') as f_test:
        f_test.write(json.dumps(test_data, indent=4))

    print("已覆盖保存 train/test 文件。")


class Instructor:
    """ Model training and evaluation """

    def __init__(self, opt):
        self.opt = opt
        if 'bert' in opt.model_name:
            tokenizer = Tokenizer4BertGCN(opt.max_length, opt.pretrained_bert_name)
            bert = BertModel.from_pretrained(opt.pretrained_bert_name)
            self.model = opt.model_class(bert, opt).to(opt.device)

            if opt.checkpoint is not None:
                # 加载预训练权重
                checkpoint = torch.load(opt.checkpoint, map_location=opt.device)

                # 检查是否包含 'state_dict' 字段（例如用 Trainer 保存的）
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                # 载入模型权重
                self.model.load_state_dict(state_dict, strict=False)  # 如果结构完全匹配可以设 strict=True

            trainset = ABSAGCNData(opt.dataset_file['train'], tokenizer, opt=opt)
            testset = ABSAGCNData(opt.dataset_file['test'], tokenizer, opt=opt)
        else:
            tokenizer = build_tokenizer(
                fnames=[opt.dataset_file['train'], opt.dataset_file['test']],
                max_length=opt.max_length,
                data_file='{}/{}_tokenizer.dat'.format(opt.vocab_dir, opt.dataset))
            embedding_matrix = build_embedding_matrix(
                vocab=tokenizer.vocab,
                embed_dim=opt.embed_dim,
                data_file='{}/{}d_{}_embedding_matrix.dat'.format(opt.vocab_dir, str(opt.embed_dim), opt.dataset))

            logger.info("Loading vocab...")
            # 词汇表类型
            token_vocab = VocabHelp.load_vocab(opt.vocab_dir + '/vocab_tok.vocab')  # token
            post_vocab = VocabHelp.load_vocab(opt.vocab_dir + '/vocab_post.vocab')  # position
            pos_vocab = VocabHelp.load_vocab(opt.vocab_dir + '/vocab_pos.vocab')  # POS
            dep_vocab = VocabHelp.load_vocab(opt.vocab_dir + '/vocab_dep.vocab')  # deprel
            pol_vocab = VocabHelp.load_vocab(opt.vocab_dir + '/vocab_pol.vocab')  # polarity
            logger.info(
                "token_vocab: {}, post_vocab: {}, pos_vocab: {}, dep_vocab: {}, pol_vocab: {}".format(len(token_vocab),
                                                                                                      len(post_vocab),
                                                                                                      len(pos_vocab),
                                                                                                      len(dep_vocab),
                                                                                                      len(pol_vocab)))

            # opt.tok_size = len(token_vocab)
            opt.post_size = len(post_vocab)
            opt.pos_size = len(pos_vocab)

            vocab_help = (post_vocab, pos_vocab, dep_vocab, pol_vocab)
            self.model = opt.model_class(embedding_matrix, opt).to(opt.device)
            trainset = SentenceDataset(opt.dataset_file['train'], tokenizer, opt=opt, vocab_help=vocab_help)
            testset = SentenceDataset(opt.dataset_file['test'], tokenizer, opt=opt, vocab_help=vocab_help)

        self.train_dataloader = DataLoader(dataset=trainset, batch_size=opt.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(dataset=testset, batch_size=opt.batch_size, shuffle=True)

        if opt.device.type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(self.opt.device.index)))
        self._print_args()

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params

        logger.info(
            'n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        logger.info('training arguments:')

        for arg in vars(self.opt):
            logger.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        for p in self.model.parameters():
            if p.requires_grad:
                if len(p.shape) > 1:
                    self.opt.initializer(p)  # xavier_uniform_
                else:
                    stdv = 1. / (p.shape[0] ** 0.5)
                    torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def get_bert_optimizer(self, model):
        # BERT 优化器配置，设置学习率和权重衰减，确保 BERT 的不同部分能以不同的学习率进行训练
        no_decay = ['bias', 'LayerNorm.weight']  # 不应用衰减
        diff_part = ["bert.embeddings", "bert.encoder"]  # 使用不同学习率

        if self.opt.diff_lr:
            logger.info("layered learning rate on")
            optimizer_grouped_parameters = [
                # Group 1: BERT 的 Embeddings/Encoder 层（带权重衰减）
                {
                    "params": [p for n, p in model.named_parameters() if
                               not any(nd in n for nd in no_decay) and any(nd in n for nd in diff_part)],
                    "weight_decay": self.opt.weight_decay,
                    "lr": self.opt.bert_lr
                },
                # Group 2: BERT 的 Embeddings/Encoder 层（不带权重衰减）
                {
                    "params": [p for n, p in model.named_parameters() if
                               any(nd in n for nd in no_decay) and any(nd in n for nd in diff_part)],
                    "weight_decay": 0.0,
                    "lr": self.opt.bert_lr
                },
                # Group 3: 其他层（带权重衰减）
                {
                    "params": [p for n, p in model.named_parameters() if
                               not any(nd in n for nd in no_decay) and not any(nd in n for nd in diff_part)],
                    "weight_decay": self.opt.weight_decay,
                    "lr": self.opt.learning_rate
                },
                # Group 4: 其他层（不带权重衰减）
                {
                    "params": [p for n, p in model.named_parameters() if
                               any(nd in n for nd in no_decay) and not any(nd in n for nd in diff_part)],
                    "weight_decay": 0.0,
                    "lr": self.opt.learning_rate
                },
            ]
            optimizer = AdamW(optimizer_grouped_parameters, eps=self.opt.adam_epsilon)

        else:
            logger.info("bert learning rate on")
            optimizer_grouped_parameters = [
                {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': self.opt.weight_decay},
                {'params': [p for n, p in model.named_parameters() if any(
                    nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            optimizer = AdamW(optimizer_grouped_parameters, lr=self.opt.bert_lr, eps=self.opt.adam_epsilon)

        return optimizer

    @staticmethod
    def contrastive_loss(representations, labels, temperature=0.5):
        """
        representations: [B, D]
        labels: [B]
        """
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        mask = labels.unsqueeze(1) == labels.unsqueeze(0)  # 正对为 1，负对为 0
        mask = mask.fill_diagonal_(False)  # 去掉自己和自己比较

        exp_sim = torch.exp(similarity_matrix / temperature)
        pos_sim = exp_sim * mask
        neg_sim = exp_sim * (~mask)

        pos_sum = pos_sim.sum(dim=1)
        neg_sum = neg_sim.sum(dim=1) + 1e-8  # 避免除 0

        loss = -torch.log(pos_sum / (pos_sum + neg_sum))
        return loss.mean()

    @staticmethod
    def info_nce_loss(representations, labels, temperature=0.5):
        """
        Supervised InfoNCE contrastive loss.

        参数:
            representations: Tensor [B, D] - 表示句子或方面的向量
            labels: Tensor [B] - 样本的情感或方面标签
            temperature: float - 温度系数，调节softmax平滑程度

        返回:
            loss: scalar - InfoNCE 对比损失
        """

        # 1. 计算余弦相似度矩阵 [B, B]
        sim_matrix = F.cosine_similarity(
            representations.unsqueeze(1),  # [B, 1, D]
            representations.unsqueeze(0),  # [1, B, D]
            dim=-1
        )  # -> [B, B]

        # 2. 除以温度，调整分布平滑性
        sim_matrix = sim_matrix / temperature

        # 3. 构建正样本掩码（标签相等且非自身）
        labels = labels.contiguous().view(-1, 1)  # [B, 1]
        mask = torch.eq(labels, labels.T).float()  # [B, B]
        mask.fill_diagonal_(0)  # 去除自对比

        # 4. softmax 分母项（包括正负样本）
        exp_sim = torch.exp(sim_matrix)  # [B, B]
        denom = exp_sim.sum(dim=1, keepdim=True)  # [B, 1]

        # 5. softmax 分子项（仅正样本）
        numerator = (exp_sim * mask).sum(dim=1)  # [B]

        # 6. 计算损失：log(正样本概率)
        # 在 denominator 添加 eps 避免 log(0)
        denom = denom + 1e-8
        numerator = numerator + 1e-8
        loss = -torch.log(numerator / denom)
        return loss.mean()

    @staticmethod
    def supervised_contrastive_loss(representations, labels, temperature=0.1):
        """
        representations: [B, D]
        labels: [B]
        """
        B = representations.size(0)
        representations = F.normalize(representations, dim=1)
        sim_matrix = torch.matmul(representations, representations.T) / temperature

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float()
        logits_mask = 1 - torch.eye(B, device=representations.device)
        mask = mask * logits_mask  # 去除自身对比

        exp_sim = torch.exp(sim_matrix) * logits_mask
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1).clamp(min=1)
        loss = -mean_log_prob_pos.mean()
        return loss

    def _train(self, criterion, optimizer, max_test_acc_overall=0):
        max_test_acc = 0
        max_f1 = 0
        global_step = 0
        model_path = ''
        for epoch in range(self.opt.num_epoch):
            logger.info('>' * 60)
            logger.info('epoch: {}'.format(epoch))
            n_correct, n_total = 0, 0
            for i_batch, sample_batched in enumerate(self.train_dataloader):
                global_step += 1
                # switch model to training mode, clear gradient accumulators
                self.model.train()
                optimizer.zero_grad()
                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                outputs, penal = self.model(inputs, contrastive=self.opt.contrastive)
                targets = sample_batched['polarity'].to(self.opt.device)
                if self.opt.contrastive:
                    # c_loss = self.supervised_contrastive_loss(penal, targets)
                    c_loss = self.info_nce_loss(penal, targets)
                    loss = criterion(outputs, targets)
                    loss += self.opt.lemda * c_loss
                else:
                    if self.opt.losstype is not None:
                        loss = criterion(outputs, targets) + penal
                    else:
                        loss = criterion(outputs, targets)

                loss.backward()
                optimizer.step()

                if global_step % self.opt.log_step == 0:
                    n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                    n_total += len(outputs)
                    train_acc = n_correct / n_total
                    test_acc, f1 = self._evaluate()
                    if test_acc > max_test_acc:
                        max_test_acc = test_acc
                        if test_acc > max_test_acc_overall:
                            if not os.path.exists('./state_dict'):
                                os.mkdir('./state_dict')
                            model_path = './state_dict/{}_{}_acc_{:.4f}_f1_{:.4f}'.format(self.opt.model_name,
                                                                                          self.opt.dataset, test_acc,
                                                                                          f1)
                            self.best_model = copy.deepcopy(self.model)
                            logger.info('>> saved: {}'.format(model_path))
                    if f1 > max_f1:
                        max_f1 = f1
                    logger.info('loss: {:.4f}, acc: {:.4f}, test_acc: {:.4f}, f1: {:.4f}'.format(loss.item(), train_acc,
                                                                                                 test_acc, f1))
        return max_test_acc, max_f1, model_path

    def _evaluate(self, show_results=False):
        # switch model to evaluation mode
        self.model.eval()
        n_test_correct, n_test_total = 0, 0
        targets_all, outputs_all = None, None
        with torch.no_grad():
            for batch, sample_batched in enumerate(self.test_dataloader):
                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                targets = sample_batched['polarity'].to(self.opt.device)
                outputs, penal = self.model(inputs)
                n_test_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                n_test_total += len(outputs)
                targets_all = torch.cat((targets_all, targets), dim=0) if targets_all is not None else targets
                outputs_all = torch.cat((outputs_all, outputs), dim=0) if outputs_all is not None else outputs
        test_acc = n_test_correct / n_test_total
        f1 = metrics.f1_score(targets_all.cpu(), torch.argmax(outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')

        labels = targets_all.data.cpu()
        predic = torch.argmax(outputs_all, -1).cpu()
        if show_results:
            report = metrics.classification_report(labels, predic, digits=4)
            confusion = metrics.confusion_matrix(labels, predic)
            return report, confusion, test_acc, f1

        return test_acc, f1

    def _test(self):
        self.model = self.best_model
        self.model.eval()
        test_report, test_confusion, acc, f1 = self._evaluate(show_results=True)
        logger.info("Precision, Recall and F1-Score...")
        logger.info(test_report)
        logger.info("Confusion Matrix...")
        logger.info(test_confusion)

    def run(self):
        criterion = nn.CrossEntropyLoss()
        if 'bert' not in self.opt.model_name:
            _params = filter(lambda p: p.requires_grad, self.model.parameters())
            optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)
        else:
            optimizer = self.get_bert_optimizer(self.model)
        max_test_acc_overall = 0
        max_f1_overall = 0
        if 'bert' not in self.opt.model_name:
            self._reset_params()

        max_test_acc, max_f1, model_path = self._train(criterion, optimizer, max_test_acc_overall)

        logger.info('max_test_acc: {0}, max_f1: {1}'.format(max_test_acc, max_f1))
        max_test_acc_overall = max(max_test_acc, max_test_acc_overall)
        max_f1_overall = max(max_f1, max_f1_overall)
        torch.save(self.best_model.state_dict(), model_path)
        logger.info('>> saved: {}'.format(model_path))
        logger.info('#' * 60)
        logger.info('max_test_acc_overall:{}'.format(max_test_acc_overall))
        logger.info('max_f1_overall:{}'.format(max_f1_overall))
        self._test()


def main():
    model_classes = {
        'otesgcn': OTESGCN,
        'ssegcnbert_ot_wo_SyntacticMask': SSEGCNBertClassifier_wo_SyntacticMask
    }

    dataset_files = {
        'restaurant': {
            'train': './dataset/Restaurants_corenlp/train_write.json',
            'test': './dataset/Restaurants_corenlp/test_write.json',
        },
        'laptop': {
            'train': './dataset/Laptops_corenlp/train_write.json',
            'test': './dataset/Laptops_corenlp/test_write.json'
        },
        'twitter': {
            'train': './dataset/Tweets_corenlp/train_write.json',
            # 'train': './dataset/Tweets_corenlp/mistake_analys.json',
            'test': './dataset/Tweets_corenlp/test_write.json',
        },
        'METS': {
            'train': './dataset/METS_corenlp/train_write.json',
            # todo：test 太长了，缩减到3000 条左右
            'test': './dataset/METS_corenlp/test_write.json',
        },
        'ASHGCN': {
            'train': './dataset/ASHGCN/Drug_corenlp/train_write.json',
            'test': './dataset/ASHGCN/Drug_corenlp/test_write.json'
        }
    }

    input_colses = {

        'ssegcn': ['text', 'aspect', 'pos', 'head', 'deprel', 'post', 'mask', 'length', 'short_mask'],
        'ssegcnbert': ['text_bert_indices', 'bert_segments_ids', 'attention_mask', 'asp_start', 'asp_end', 'src_mask',
                       'aspect_mask', 'short_mask'],
        'ssegcnbert_ot': ['text_bert_indices', 'bert_segments_ids', 'attention_mask', 'asp_start', 'asp_end',
                          'src_mask', 'aspect_mask', 'short_mask'],
        'ssegcnbert_ot_2': ['text_bert_indices', 'bert_segments_ids', 'attention_mask', 'asp_start', 'asp_end',
                            'src_mask', 'aspect_mask', 'short_mask'],
        'ssegcnbert_ot_wo_SyntacticMask': ['text_bert_indices', 'bert_segments_ids', 'attention_mask', 'asp_start',
                                           'asp_end', 'src_mask', 'aspect_mask', 'short_mask']
    }

    # 设定神经网络层的初始权重，包括均匀分布、正态分布和正交矩阵，保持向量的正交性
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }

    # 优化器
    optimizers = {
        'adadelta': torch.optim.Adadelta,
        'adagrad': torch.optim.Adagrad,
        'adam': torch.optim.Adam,
        'adamax': torch.optim.Adamax,
        'asgd': torch.optim.ASGD,
        'rmsprop': torch.optim.RMSprop,
        'sgd': torch.optim.SGD,
    }

    # Hyperparameters
    parser = argparse.ArgumentParser()
    # 模型，包括 otesgcn
    # 消融：ssegcnbert_ot_wo_SyntacticMask
    parser.add_argument('--model_name', default='otesgcn', type=str,
                        help=', '.join(model_classes.keys()))
    # 比如：restaurant，laptop，twitter，METS, ASHGCN
    parser.add_argument('--dataset', default='ASHGCN', type=str, help=', '.join(dataset_files.keys()))
    # 是否使用对比学习
    parser.add_argument('--contrastive', default=True, type=bool)
    # 对比学习损失的权重
    parser.add_argument('--lemda', default=0.8, type=float)
    # short 矩阵的 dropout
    parser.add_argument('--learning_rate', default=0.002, type=float)  # 0.002
    parser.add_argument('--short_drop', default=0.1, type=float, help='short矩阵的drop概率')
    parser.add_argument('--alpha', default=1.0, type=float, help='图注意力和OT注意力的权重')
    parser.add_argument('--num_layers', type=int, default=6, help='Num of attn layers.')
    parser.add_argument('--attdim', type=int, default=200, help='注意力的维度')
    parser.add_argument('--num_epoch', default=10, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--attn_dropout', type=float, default=1.0, help='Attn layer dropout rate.')
    parser.add_argument("--ot_epsilon", default=2, type=float, help="Epsilon for OT-Attention.")
    # max_length 要设大一点，不然会报错
    parser.add_argument('--max_length', default=128, type=int)
    parser.add_argument('--device', default=None, type=str, help='cpu, cuda')
    parser.add_argument('--split_data', default=False, type=bool, help='是否重新分配数据')
    parser.add_argument('--checkpoint', default=None, type=str, help='预训练模型权重')

    parser.add_argument('--optimizer', default='adam', type=str, help=', '.join(optimizers.keys()))
    parser.add_argument('--initializer', default='xavier_uniform_', type=str, help=', '.join(initializers.keys()))
    parser.add_argument('--l2reg', default=2e-5, type=float)  # 1e-4
    # 分类的类别
    parser.add_argument('--polarities_dim', default=3, type=int, help='3')
    parser.add_argument('--log_step', default=5, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--post_dim', type=int, default=30, help='Position embedding dimension.')
    parser.add_argument('--pos_dim', type=int, default=30, help='Pos embedding dimension.')
    parser.add_argument('--hidden_dim', type=int, default=50, help='GCN mem dim.')  # 50

    parser.add_argument('--input_dropout', type=float, default=0.7, help='Input dropout rate.')
    parser.add_argument('--lower', default=True, help='Lowercase all words.')
    parser.add_argument('--direct', default=False, help='directed graph or undirected graph')
    parser.add_argument('--loop', default=True)

    parser.add_argument('--bidirect', default=True, help='Do use bi-RNN layer.')
    parser.add_argument('--rnn_hidden', type=int, default=50, help='RNN hidden state size.')
    parser.add_argument('--rnn_layers', type=int, default=1, help='Number of RNN layers.')
    parser.add_argument('--rnn_dropout', type=float, default=0.1, help='RNN dropout rate.')

    parser.add_argument('--attention_heads', default=5, type=int, help='number of multi-attention heads')
    parser.add_argument('--seed', default=1000, type=int)
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
    parser.add_argument('--vocab_dir', type=str, default='./dataset/Laptops_corenlp')
    parser.add_argument('--pad_id', default=0, type=int)
    parser.add_argument('--parseadj', default=False, action='store_true', help='dependency probability')
    parser.add_argument('--parsehead', default=False, action='store_true', help='dependency tree')
    parser.add_argument('--cuda', default='0', type=str)
    parser.add_argument('--losstype', default=None, type=str,
                        help="['doubleloss', 'orthogonalloss', 'differentiatedloss']")
    parser.add_argument('--beta', default=0.25, type=float)

    # * bert
    parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument('--bert_dim', type=int, default=768)
    parser.add_argument('--bert_dropout', type=float, default=0.3, help='BERT dropout rate.')
    parser.add_argument('--diff_lr', default=True, action='store_true')
    parser.add_argument('--bert_lr', default=2e-5, type=float)
    opt = parser.parse_args()

    opt.model_class = model_classes[opt.model_name]
    opt.dataset_file = dataset_files[opt.dataset]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]

    print("choice cuda:{}".format(opt.cuda))
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if opt.device is None else torch.device(
        opt.device)

    # 重新读取和分配数据
    if opt.split_data:
        read_data_split(opt)

    # set random seed
    setup_seed(opt.seed)

    if not os.path.exists('./log'):
        os.makedirs('./log', mode=0o777)
    #  todo：需要将文件名中的冒号改为下划线_
    log_file = '{}-{}-{}.log'.format(opt.model_name, opt.dataset, strftime("%Y-%m-%d_%H_%M_%S", localtime()))
    logger.addHandler(logging.FileHandler("%s/%s" % ('./log', log_file)))

    ins = Instructor(opt)
    ins.run()


if __name__ == '__main__':
    main()
