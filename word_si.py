import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import pkuseg
from collections import Counter
import time
from torch.utils.data import TensorDataset
import torch.utils.data as Data
from torch.optim import AdamW
import random
import torch.nn.functional as F


def seed_init(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_dataloader():
    Batch_Size = 32

    data = pd.read_csv('./data/word_similarity.csv')
    data = data[:20000]
    # data = data[:2000]
    seg = pkuseg.pkuseg()

    data['t1'] = data["sen_a"].apply(lambda x: list(seg.cut(x)))
    data['t2'] = data["sen_b"].apply(lambda x: list(seg.cut(x)))
    # print(data.head())
    # 生成词语表
    with open("./data/vocab.txt", 'w', encoding='utf-8') as fout:
        fout.write("<unk>\n")
        fout.write("<pad>\n")
        # 使用 < unk > 代表未知字符且将出现次数为1的作为未知字符
        # 实用 < pad > 代表需要padding的字符(句子长度进行统一)
        vocab1 = [word for word, freq in Counter(j for i in data['t1'] for j in i).most_common() if freq > 1]
        vocab2 = [word for word, freq in Counter(j for i in data['t2'] for j in i).most_common() if freq > 1]
        vocab = set(vocab1 + vocab2)
        # print(vocab)
        for i in vocab:
            fout.write(i + "\n")

    # 初始化生成 词对序 与 序对词 表
    with open("./data/vocab.txt", encoding='utf-8') as fin:
        vocab = [i.strip() for i in fin]
    char2idx = {i: index for index, i in enumerate(vocab)}
    idx2char = {index: i for index, i in enumerate(vocab)}
    vocab_size = len(vocab)
    pad_id = char2idx["<pad>"]
    unk_id = char2idx["<unk>"]

    Max_length = 18

    def tokenizer(name):
        inputs = []
        sentence_char = [[j for j in i] for i in data[name]]
        # 将输入文本进行padding
        for index, i in enumerate(sentence_char):
            temp = [char2idx.get(j, unk_id) for j in i]
            if len(temp) < Max_length:
                for _ in range(Max_length - len(temp)):
                    temp.append(pad_id)
            else:
                temp = temp[:Max_length]
            inputs.append(temp)
        return inputs

    class TextCNNDataSet(Data.Dataset):
        def __init__(self, data_1, data_2, data_targets):
            self.t1 = torch.LongTensor(data_1)
            self.t2 = torch.LongTensor(data_2)
            self.label = torch.LongTensor(data_targets)

        def __getitem__(self, index):
            return self.t1[index], self.t2[index], self.label[index]

        def __len__(self):
            return len(self.label)

    TextCNNDataSet = TextCNNDataSet(tokenizer('t1'), tokenizer('t2'), list(data["label"]))
    train_size = int(len(list(data["label"])) * 0.8)
    test_size = len(list(data["label"])) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(TextCNNDataSet, [train_size, test_size])

    test_size = int(len(list(test_dataset)) * 0.5)
    eval_size = int(len(list(test_dataset))) - test_size
    eval_dataset, test_dataset = torch.utils.data.random_split(test_dataset, [eval_size,test_size])

    TrainDataLoader = Data.DataLoader(train_dataset, batch_size=Batch_Size, shuffle=True, drop_last=True)
    EvalDataLoader = Data.DataLoader(eval_dataset, batch_size=Batch_Size, shuffle=False, drop_last=True)
    TestDataLoader = Data.DataLoader(test_dataset, batch_size=Batch_Size, shuffle=False, drop_last=True)
    print('Training {} samples...'.format(len(TrainDataLoader.dataset)))
    print('Evading {} samples...'.format(len(EvalDataLoader.dataset)))
    print('Testing {} samples...'.format(len(TestDataLoader.dataset)))
    return TrainDataLoader, EvalDataLoader, TestDataLoader


class same_con(nn.Module):
    def __init__(self):
        super(same_con, self).__init__()
        self.Vocab_size = 14331  # 词典大小
        # self.Vocab_size = 2400
        self.batch_size = 32
        self.n_hidden1 = 64  # bi-lstm的隐藏层大小
        self.Embedding_dim = 256  # 词嵌入维度，这里我使用了word2vec预训练的词向量
        self.n_class = 2  # 相似和不相似两种分类
        self.dropout = nn.Dropout(0.5)  # dropout设置为0.5，不知道后面起没起作用
        # self.Embedding_matrix = Embedding_matrix  # 词嵌入矩阵，#size=[Vocab_size,embedding_size]，可以自己训练一个词向量矩阵。
        self.word_embeds = nn.Embedding(self.Vocab_size, self.Embedding_dim)  # 嵌入层
        # pretrained_weight = np.array(self.Embedding_matrix)  # 转换成numpy类型
        # self.word_embeds.weight.data.copy_(torch.from_numpy(pretrained_weight))  # 将词嵌入矩阵放到网络结构中
        self.Bi_Lstm1 = nn.LSTM(self.Embedding_dim, hidden_size=self.n_hidden1,
                                bidirectional=True)  # bi-lstm,hidden_size = 128

        self.fc = nn.Linear(self.n_hidden1 * 2, self.n_class, bias=False)  # 根据attention后的输出，全连接层的大小设置为(256,2)
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

    def forward(self, train_left, train_right):
        train_left = self.word_embeds(train_left).to(device)  # train_left为索引，mapping词向量，得到一个词向量矩阵，size(12,300)
        train_right = self.word_embeds(train_right).to(device)  # 同上

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


def train(model, device, tloader, eloader, optimizer, criterion, epochs, path):
    now_cnt, stop_cnt = 0, 10
    best_acc = 0

    for epoch in range(epochs):
        # 训练--------------------------------------
        t_a = time.time()

        model.train()
        train_loss = 0
        num_correct = 0
        for batch_idx, batch in enumerate(tloader):  # 返回的是元组，记得加括号
            train_left = batch[0].to(device)
            train_right = batch[1].to(device)
            lables = batch[2].to(device)

            optimizer.zero_grad()
            output = model(train_left, train_right)
            loss = criterion(output, lables)
            loss.backward()
            optimizer.step()

            train_loss += float(loss.item())
            true = lables.data.cpu()
            res = torch.max(output, dim=1)[1].cpu()
            num_correct += torch.eq(res, true).sum().float().item()
        train_acc = num_correct / len(tloader.dataset)
        train_loss = train_loss / len(tloader)
        t_b = time.time()
        msg = 'Epoch: {0:>5},  Train Loss: {1:>5.5},  Train Acc: {2:>6.4%} ,Time: {3:>6.5}'
        print(msg.format(epoch, train_loss, train_acc, t_b - t_a))

        # 验证--------------------------------------
        t_a = time.time()
        model.eval()
        eval_loss = 0
        num_correct = 0
        for batch_idx, batch in enumerate(eloader):
            with torch.no_grad():
                eval_left = batch[0].to(device)
                eval_right = batch[1].to(device)
                lables = batch[2].to(device)

                output = model(eval_left, eval_right)
                loss = criterion(output, lables)

                eval_loss += float(loss.item())
                true = lables.data.cpu()
                res = torch.max(output, dim=1)[1].cpu()
                num_correct += torch.eq(res, true).sum().float().item()
        eval_acc = num_correct / len(eloader.dataset)
        eval_loss = eval_loss / len(eloader)
        t_b = time.time()
        msg = 'Epoch: {0:>5},  eval Loss: {1:>5.5},  eval Acc: {2:>6.4%} ,Time: {3:>6.5}'
        print(msg.format(epoch, eval_loss, eval_acc, t_b - t_a))

        if eval_acc > best_acc:
            print('训练效果更好，保存模型参数')
            best_acc = eval_acc
            torch.save(model.state_dict(), path)
            now_cnt = 0
        else:
            now_cnt += 1
            if now_cnt > stop_cnt:
                print('模型已无提升停止训练,验证集最高精度:', best_acc)
                break


def predict(model, device, loader):
    t_a = time.time()
    model.eval()
    num_correct = 0
    for batch_idx, batch in enumerate(loader):
        with torch.no_grad():  # 返回的是元组，记得加括号
            test_left = batch[0].to(device)
            test_right = batch[1].to(device)
            lables = batch[2].to(device)

            output = model(test_left, test_right)
            true = lables.data.cpu()
            res = torch.max(output, dim=1)[1].cpu()
            num_correct += torch.eq(res, true).sum().float().item()
    test_acc = num_correct / len(loader.dataset)
    t_b = time.time()
    msg = 'Test Acc: {0:>6.4%} ,Time: {1:>6.5}'
    print(msg.format(test_acc, t_b - t_a))


if __name__ == '__main__':
    SEED = 721
    Learn_rate = 1e-3
    Epochs = 5
    save_path = 'model/use_att.pkl'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    seed_init(SEED)

    train_loader, eval_loader, test_loader = get_dataloader()
    Bi_LstmModel = same_con().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = AdamW(Bi_LstmModel.parameters(), lr=Learn_rate)

    train(Bi_LstmModel, device, train_loader, eval_loader, optimizer, criterion, Epochs + 1, save_path)

    Bi_LstmModel.load_state_dict(torch.load(save_path))
    predict(Bi_LstmModel, device, test_loader)
