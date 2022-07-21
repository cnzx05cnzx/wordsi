import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import time
from torch.utils.data import TensorDataset, DataLoader
import torch.utils.data as Data
from torch.optim import AdamW
import random
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup

from sklearn import metrics


def seed_init(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def preprocessing_for_bert(text, toker, MaxLen):
    encoded_sent = toker.encode_plus(
        text=text,  # 预处理语句
        add_special_tokens=True,  # 加 [CLS] 和 [SEP]
        max_length=MaxLen,  # 截断或者填充的最大长度
        padding='max_length',  # 填充为最大长度，这里的padding在之间可以直接用pad_to_max但是版本更新之后弃用了，老版本什么都没有，可以尝试用extend方法
        return_attention_mask=True,  # 返回 attention mask
        truncation=True
    )

    # 把list转换为tensor
    # input_ids = torch.LongTensor(input_ids)
    # attention_masks = torch.LongTensor(attention_masks)

    return encoded_sent.get('input_ids'), encoded_sent.get('attention_mask')


def get_dataloader(batch_size, toker):
    data_train = pd.read_csv('./data/medicial/train.csv', encoding='gbk')
    data_eval = pd.read_csv('./data/medicial/dev.csv', encoding='gbk')
    data_test = pd.read_csv('./data/medicial/test.csv', encoding='gbk')
    # data = data[:20000]
    # data = data[:2000]

    Max_Length = 15

    def deal_text(data, name):
        inputs = []
        masks = []
        sentence_char = [i for i in data[name]]
        # 将输入文本进行padding
        for index, i in enumerate(sentence_char):
            a, b = preprocessing_for_bert(i, toker, Max_Length)
            inputs.append(a)
            masks.append(b)

        return inputs, masks

    class TextDataSet(Data.Dataset):
        def __init__(self, data1, data2, data_targets):
            self.t1i = torch.LongTensor(data1[0])
            self.t1m = torch.LongTensor(data1[1])
            self.t2i = torch.LongTensor(data2[0])
            self.t2m = torch.LongTensor(data2[1])
            self.label = torch.LongTensor(data_targets)

        def __getitem__(self, index):
            return self.t1i[index], self.t1m[index], self.t2i[index], self.t2m[index], self.label[index]

        def __len__(self):
            return len(self.label)

    # res = deal_text(data_train, 'q1')
    train_dataset = TextDataSet(deal_text(data_train, 'q1'), deal_text(data_train, 'q2'), list(data_train["label"]))
    train_dataloader = DataLoader(train_dataset, shuffle=True, drop_last=True, batch_size=batch_size)
    eval_dataset = TextDataSet(deal_text(data_eval, 'q1'), deal_text(data_eval, 'q2'), list(data_eval["label"]))
    eval_dataloader = DataLoader(eval_dataset, shuffle=False, drop_last=True, batch_size=batch_size)
    test_dataset = TextDataSet(deal_text(data_test, 'q1'), deal_text(data_test, 'q2'), list(data_test["label"]))
    test_dataloader = DataLoader(test_dataset, shuffle=False, drop_last=True, batch_size=batch_size)

    return train_dataloader, eval_dataloader, test_dataloader


class SameConBert(nn.Module):
    def __init__(self, batch_size, freeze_bert=False):
        super(SameConBert, self).__init__()
        self.batch_size = batch_size
        self.n_hidden1 = 64  # bi-lstm的隐藏层大小
        self.Embedding_dim = 768  # 词嵌入维度，这里我使用了word2vec预训练的词向量
        self.n_class = 3  # 相似和不相似两种分类
        self.dropout = nn.Dropout(0.3)  # dropout设置为0.5，不知道后面起没起作用
        self.bert = BertModel.from_pretrained(model_choice, return_dict=False)
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.Bi_Lstm1 = nn.LSTM(self.Embedding_dim, hidden_size=self.n_hidden1,
                                bidirectional=True)  # bi-lstm,hidden_size = 128

        self.fc = nn.Linear(self.n_hidden1 * 2, self.n_class)
        self.b = nn.Parameter(torch.rand([self.n_class]))  # 偏置b
        pass

    def attention_weight1(self, outputs1, final_state1):  # 最好debug一步一步看怎么变化
        outputs1 = outputs1.permute(1, 0, 2)
        hidden = final_state1.view(-1, self.n_hidden1 * 2, 1)
        attention_weights = torch.bmm(outputs1, hidden).squeeze(2)
        # z=torch.bmm(x,y)针对三维数组相乘，x=[a,b,c],y =[a,c,d], z = [a,b,c],这里的a,b,c,d都是size.
        soft_attention_weights1 = F.softmax(attention_weights, 1)
        context1 = torch.bmm(outputs1.transpose(1, 2), soft_attention_weights1.unsqueeze(2)).squeeze(2)
        return context1, soft_attention_weights1
        pass

    def forward(self, lefti, lefto, righti, righto):
        train_left, _ = self.bert(input_ids=lefti, attention_mask=lefto)
        train_right, _ = self.bert(input_ids=righti, attention_mask=righto)
        train_left = train_left.to(device)
        train_right = train_right.to(device)

        train_left = train_left.transpose(0, 1)  # 交换维度 ，变为[seq_len,batch_size,embedding_dim]
        train_right = train_right.transpose(0, 1)

        hidden_state1 = torch.rand(2, self.batch_size, self.n_hidden1).to(device)  # 隐藏层单元初始化
        cell_state1 = torch.rand(2, self.batch_size, self.n_hidden1).to(device)

        outputs1_L, (final_state1_L, _) = self.Bi_Lstm1(train_left, (hidden_state1, cell_state1))  # 左右两边参数共享
        outputs1_L = self.dropout(outputs1_L)  # 左边输出
        attn_outputs1_L, attention1_L = self.attention_weight1(outputs1_L, final_state1_L)  # 左右两边attention也共享

        outputs1_R, (final_state1_R, _) = self.Bi_Lstm1(train_right, (hidden_state1, cell_state1))
        outputs1_R = self.dropout(outputs1_R)
        attn_outputs1_R, attention1_R = self.attention_weight1(outputs1_R, final_state1_R)

        outputs1 = attn_outputs1_L  # attention后的输出
        outputs2 = attn_outputs1_R

        output = torch.abs(outputs1 - outputs2)  # 采用的是曼哈顿距离（两个向量相减的绝对值），也可以使用别的距离
        output = self.fc(output) + self.b  # 全连接，得到二分类
        output = F.softmax(output, dim=1)  # softmax函数归一
        return output
        pass


def initialize_model(loader, batch_size, epochs):
    """
    初始化我们的bert，优化器还有学习率，epochs就是训练次数
    """
    # 初始化我们的Bert分类器
    bert_classifier = SameConBert(batch_size)
    # 用GPU运算
    bert_classifier.to(device)
    # 创建优化器
    optimizer = AdamW(bert_classifier.parameters(),
                      lr=5e-5,
                      eps=1e-8
                      )
    total_steps = len(loader) * epochs
    warm_ratio = 0
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warm_ratio * total_steps,
                                                num_training_steps=total_steps)
    criterion = nn.CrossEntropyLoss()

    return bert_classifier, optimizer, criterion, scheduler


def train(model, train_dataloader, valid_dataloader, optimizer, criterion, scheduler, epochs, path):
    best_acc = 0
    for epoch_i in range(epochs):
        print("epoch %d" % (epoch_i + 1))

        time_begin = time.time()
        train_loss = []

        model.train()

        for step, batch in enumerate(train_dataloader):
            # print(batch)
            data1i, data1m, data2i, data2m, labels = tuple(t.to(device) for t in batch)
            model.zero_grad()
            logits = model(data1i, data1m, data2i, data2m)
            loss = criterion(logits, labels)
            train_loss.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

        time_elapsed = time.time() - time_begin
        avg_train_loss = sum(train_loss) / len(train_loss)
        print("训练集 Loss: %.2f 时间: %.2f" % (avg_train_loss, time_elapsed))

        # ---------------------------------------验证------------------------------
        model.eval()
        valid_accuracy = []
        valid_loss = []

        # 用于早停机制（这里由于epochs过少，不使用）
        # cnt, stop = 0, 10

        time_begin = time.time()
        y_pred, y_true = [], []
        for step, batch in enumerate(valid_dataloader):
            with torch.no_grad():
                data1i, data1m, data2i, data2m, labels = tuple(t.to(device) for t in batch)
                logits = model(data1i, data1m, data2i, data2m)
                loss = criterion(logits, labels)
                valid_loss.append(loss.item())

                preds = torch.argmax(logits, dim=1).flatten()

                accuracy = (preds == labels).cpu().numpy().mean() * 100

                valid_accuracy.append(accuracy)

                y_pred.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())
                y_true.extend(labels.cpu().numpy().tolist())

        time_elapsed = time.time() - time_begin
        valid_f1 = metrics.f1_score(y_true, y_pred, average='macro')
        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_accuracy = sum(valid_accuracy) / len(valid_accuracy)
        print("验证集 Loss: %.2f 验证集 Acc: %.2f f1: %.2f 时间: %.2f" % (valid_loss, valid_accuracy, valid_f1, time_elapsed))
        # 采用准确率作为评价指标
        if best_acc < valid_accuracy:
            # best_loss = valid_loss
            best_acc = valid_accuracy
            torch.save(model.state_dict(), path)  # save entire net

            print('保存最好效果模型')

        print("\n")
    print("验证集最高准确率为%.2f" % best_acc)


if __name__ == '__main__':
    SEED = 721
    Learn_rate = 1e-3
    Epochs = 60
    BatchSize = 64
    save_path = 'model/use_att.pkl'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    seed_init(SEED)

    model_choice = './hfl/chinese-bert-wwm'
    tokenizer = BertTokenizer.from_pretrained(model_choice)

    train_loader, eval_loader, test_loader = get_dataloader(BatchSize, tokenizer)
    net, optimizer, criterion, scheduler = initialize_model(train_loader, BatchSize, Epochs)
    save_path = 'model/bert_params.pkl'
    print("Start training and validating:")
    train(net, train_loader, eval_loader, optimizer, criterion, scheduler, Epochs, save_path)
