import torch
import random
import collections
import os
import tarfile
from torch import nn, optim
import torchtext.vocab as Vocab
import torch.utils.data as Data
import time
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = "../Datasets/"

# 读取数据
filename = os.path.join(DATA_ROOT, "aclImdb.tar.gz")
if not os.path.exists(os.path.join(DATA_ROOT, "aclImdb")):
    print(" 从压缩包解压...")
    with tarfile.open(filename, "r") as f:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(f, DATA_ROOT)


def read_imdb(folder="train", data_root=DATA_ROOT+"/aclImdb"):
    data = []
    for label in ["pos", "neg"]:
        folder_name = os.path.join(data_root, folder, label)
        print("loading data...")
        for file in tqdm(os.listdir(folder_name)):
            with open(os.path.join(folder_name, file), "r", encoding="utf-8") as f:
                content = f.read().replace("\n", " ").lower()
                data.append([content, 1 if label == "pos" else 0])
    random.shuffle(data)
    return data


train_data, test_data = read_imdb("train"), read_imdb("test")


# 数据预处理
def get_tokenized_imdb(data):
    '''
    :param data: list of [string, label]
    :return: 对每个句子分词，shape=[[],[]...]
    '''
    def tokenizer(text):
        return [tok.lower() for tok in text.split()]
    return [tokenizer(review) for review, label in data]


def get_vocab_imdb(data):
    tokenized_data = get_tokenized_imdb(data)
    counter = collections.Counter([tk for st in tokenized_data for tk in st])
    return Vocab.Vocab(counter, min_freq=5)


vocab = get_vocab_imdb(train_data)


def preprocess_imdb(data, vocab):
    max_l = 500
    def pad(x):
        return x[:max_l] if len(x)>max_l else x+[0]*(max_l-len(x))
    tokenized_data = get_tokenized_imdb(data)
    features = torch.tensor([pad([vocab.stoi[word] for word in words]) for words in tokenized_data])
    labels = torch.tensor([label for review, label in data])
    return features, labels


# 创建数据迭代器
batch_size = 64
train_set = Data.TensorDataset(*preprocess_imdb(train_data, vocab))
train_iter = Data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

test_set = Data.TensorDataset(*preprocess_imdb(test_data, vocab))
test_iter = Data.DataLoader(train_set, batch_size=batch_size)

# 测试
# for x, y in train_iter:
#     print("x:", x.shape, "\ny:", y.shape)
#     break
# print(len(train_iter))


# 定义模型
class BiRNN(nn.Module):
    def __init__(self, embed_size, num_hiddens, num_layers):
        super(BiRNN, self).__init__()
        self.embedding = nn.Embedding(len(vocab), embed_size)
        self.encoder = nn.LSTM(input_size=embed_size,
                               hidden_size= num_hiddens,
                               num_layers= num_layers,
                               bidirectional=True)
        self.decoder = nn.Linear(4*num_hiddens, 2)

    def forward(self, inputs):
        # inputs.shape = [batch_size, max_L], lstm是按序列顺序输入，因此需要将inputs转置
        embeddings = self.embedding(inputs.permute(1, 0))   # shape = [max_l, batch_size, embed_dim]
        outputs, _ = self.encoder(embeddings)   # outputs.shape=[max_l, batch_size, 2*num_hiddens]
        encoding = torch.cat((outputs[0], outputs[-1]), -1)
        outs = self.decoder(encoding)
        return outs

embed_size, num_hiddens, num_layers = 100, 100, 2
net = BiRNN(embed_size, num_hiddens, num_layers)


# 加载预训练的词向量
cache = "D://Desktop/Admin/NLP/dataset/GlovePretrainWordvec"
glove_vocab = Vocab.GloVe(name="6B", dim=100, cache=cache)
# print(glove_vocab.vectors.shape)
# print(glove_vocab.vectors[0].shape[0])


# 从预训练好的vocab中提取words的对应向量
def load_pretrained_embedding(words, pretrained_vocab):
    embed = torch.zeros(len(words), pretrained_vocab.vectors[0].shape[0])
    oov_count = 0
    for i, word in enumerate(words):
        try:
            idx = pretrained_vocab.stoi[word]
            embed[i, :] = pretrained_vocab.vectors[idx]
        except KeyError:
            oov_count += 1
    if oov_count > 0:
        print("有%d个单词表以外的单词"%oov_count)
    return embed


net.embedding.weight.data.copy_(load_pretrained_embedding(vocab.itos, glove_vocab))
net.embedding.weight.requires_grad = False   #  直接加载预训练好的, 所以不需要更新它


def evaluate_accuracy(test_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        device = list(net.parameters())[0].device
    test_acc_sum, n= 0.0, 0
    with torch.no_grad():
        for x, y in test_iter:
            x, y = x.to(device), y.to(device)
            if isinstance(net, torch.nn.Module):
                net.eval()
                y_hat = net(x)
                test_acc_sum += (y_hat.argmax(dim=1)==y).sum().cpu().item()
                net.train()
            else:
                if "is_training" in net.__code__.co_varnames:
                    y_hat = net(x,is_training=False)
                    test_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
                else:
                    y_hat = net(x)
                    test_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += x.shape[0]
        return test_acc_sum/n



def train(train_iter, test_iter, net, loss, optimizer, device, num_epochs):
    net = net.to(device)
    print("train on ", device)
    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for x, y in train_iter:
            x = x.to(device)
            y = y.to(device)
            y_hat = net(x)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l
            train_acc_sum += (y_hat.argmax(dim=1) == y).float().sum().cpu().item()
            n += x.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print("epoch: %d  train_l_sum: %.3f  train_acc_sum: %.3f  test_acc: %.3f  used time: %.3f"%(epoch+1,
            train_l_sum/batch_size, train_acc_sum/n, test_acc, time.time()-start))


def predict_sentiment(net, vocab, sentence):
    device = list(net.parameters())[0].device
    sentence = torch.tensor([vocab.stoi[word] for word in sentence], device=device)
    label = torch.argmax(net(sentence.view((1, -1))), dim=1)
    return "positive" if label.item() == 1 else "negative"



# 训练模型
lr, num_epochs = 0.01, 5
optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, net.parameters()), lr=lr)
loss = nn.CrossEntropyLoss()
train(train_iter, test_iter, net, loss, optimizer, device, num_epochs)

sentence = [["this", "movie", "is", "so", "great"], ["this", "movie", "is", "not", "good"]]
for st in sentence:
    result = predict_sentiment(net, vocab, st)
    print("the sentence【" + " ".join(st) + "】 is【%s】"%result)





