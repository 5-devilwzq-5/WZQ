import math
import time
from datetime import datetime
from torch.utils.data import Subset
import torch
from einops import rearrange
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchtext import datasets
from transformers import BertTokenizerFast


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, device=None):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        self.register_buffer('positional_encoding', self.encoding)

        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)
        self.encoding = self.encoding.to(device)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1)]


class AttentionMulti(nn.Module):
    def __init__(self, embedding_feature_dim, attention_feature_dim, heads, dropout=0.):
        super(AttentionMulti, self).__init__()
        self.w_qkv = nn.Linear(embedding_feature_dim, attention_feature_dim * heads * 3)
        self.heads = heads
        self.dk = attention_feature_dim / heads
        self.attention_to_embedding = nn.Linear(attention_feature_dim * heads, embedding_feature_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        qkv = self.w_qkv(x)
        qkv = qkv.chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        attention_tensor = torch.matmul(q, k.transpose(-1, -2))
        attention_tensor = attention_tensor / torch.sqrt(torch.tensor(self.dk, dtype=torch.float32))

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            attention_tensor = attention_tensor.masked_fill(mask == 0, float('-inf'))

        attention_tensor = attention_tensor.softmax(dim=-1)
        attention_tensor = self.dropout(attention_tensor)

        out = torch.matmul(attention_tensor, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h=self.heads)
        out = self.attention_to_embedding(out)
        out = self.dropout(out)
        return out


class Transformer(nn.Module):
    def __init__(self, depth, embedding_feature_dim, attention_feature_dim, heads, mlp_feature_dim, dropout=0.):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            attention = AttentionMulti(embedding_feature_dim, attention_feature_dim, heads)
            mlp = nn.Sequential(
                nn.LayerNorm(embedding_feature_dim),
                nn.Linear(embedding_feature_dim, mlp_feature_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_feature_dim, embedding_feature_dim),
                nn.Dropout(dropout),
            )
            self.layers.append(nn.ModuleList([attention, mlp]))
        self.norm = nn.LayerNorm(embedding_feature_dim)

    def forward(self, x, mask):
        for attention, mlp in self.layers:
            x = x + attention(x, mask)
            x = x + mlp(x)
        return self.norm(x)


class TextClassifyTransformer(nn.Module):
    def __init__(self, class_num, vocab_size, sequence_length, depth, embedding_feature_dim, attention_feature_dim,
                 heads, mlp_feature_dim, dropout=0., emb_dropout=0., device=None):
        super(TextClassifyTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_feature_dim)
        self.ln1 = nn.LayerNorm(embedding_feature_dim)
        self.pos_encoder = PositionalEncoding(d_model=embedding_feature_dim, max_len=sequence_length, device=device)
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(depth, embedding_feature_dim, attention_feature_dim, heads, mlp_feature_dim,
                                       dropout)
        self.fc = nn.Linear(embedding_feature_dim, class_num)

    def forward(self, x, mask):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.emb_dropout(x)
        x = self.ln1(x)
        x = self.transformer(x, None)
        x = x.mean(dim=1)
        return self.fc(x)


class TextClassify:
    def __init__(self, train_data_path=None, workers=8,
                 batch_size=128, epochs_num=2, lr=1e-5, target_path="./target"):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"use divice:{self.device}")
        self.writer = SummaryWriter(log_dir='./log/' + time.strftime('%m-%d_%H.%M', time.localtime()))
        self.target_path = target_path
        self.batch_size = batch_size
        self.epochs_num = epochs_num
        self.lr = lr
        self.workers = workers
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        print(f"The vocabulary size is: {self.tokenizer.vocab_size}")
        self.model = TextClassifyTransformer(4, self.tokenizer.vocab_size, sequence_length=512, depth=6,
                                             embedding_feature_dim=300, attention_feature_dim=128, heads=8,
                                             mlp_feature_dim=1200, dropout=0.1, emb_dropout=0.1, device=self.device)
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-2)
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.7)
        self.train_data_loader = None
        self.valid_data_loader = None
        self.train_data_size = None
        self.valid_data_size = None
        self.valid_batch_index = 0
        self.train_batch_index = 0
        self.__load_data(train_data_path)

    def __load_data(self, train_data_path):
        train_data = datasets.AG_NEWS(root=train_data_path, split='train')
        valid_data = datasets.AG_NEWS(root=train_data_path, split='test')
        train_list = []
        for i, example in enumerate(train_data):
            if i >= 1200:
                break
            train_list.append(example)

        valid_list = []
        for i, example in enumerate(valid_data):
            if i >= 76:
                break
            valid_list.append(example)

        train_labels, train_texts = zip(*train_list)
        valid_labels, valid_texts = zip(*valid_list)

        class TextDataset(torch.utils.data.Dataset):
            def __init__(self, labels, texts):
                self.labels = labels
                self.texts = texts

            def __getitem__(self, index):
                return self.labels[index], self.texts[index]

            def __len__(self):
                return len(self.labels)

        train_data = TextDataset(train_labels, train_texts)
        valid_data = TextDataset(valid_labels, valid_texts)

        self.train_data_loader = DataLoader(dataset=train_data, batch_size=self.batch_size, shuffle=True)
        self.valid_data_loader = DataLoader(dataset=valid_data, batch_size=self.batch_size, shuffle=True)
        self.train_data_size = 1200
        self.valid_data_size = 76

    def train_model(self):
        print("start train model...")
        print(f"hyper-parameters: batch_size:{self.batch_size}; "
              f"epochs_num:{self.epochs_num}; lr:{self.lr};")

        best_epoch_acc_rate_valid = 0

        self.train_batch_index = 0
        self.valid_batch_index = 0

        for epoch in range(self.epochs_num):
            epoch_loss_train, epoch_acc_rate_train = self.__do_train(epoch)
            self.lr_scheduler.step()
            epoch_loss_valid, epoch_acc_rate_valid = self.__do_valid(epoch)

            self.writer.add_scalars("epoch_loss", {"train": epoch_loss_train}, epoch)
            self.writer.add_scalars("epoch_loss", {"valid": epoch_loss_valid}, epoch)
            self.writer.add_scalars("epoch_acc", {"train": epoch_acc_rate_train}, epoch)
            self.writer.add_scalars("epoch_acc", {"valid": epoch_acc_rate_valid}, epoch)

            print(f"epoch {epoch}/{self.epochs_num - 1} : "
                  f"epoch_loss_train:{epoch_loss_train:.4f}; epoch_acc_rate_train:{epoch_acc_rate_train:.4f}; "
                  f"epoch_loss_valid:{epoch_loss_valid:.4f}; epoch_acc_rate_valid:{epoch_acc_rate_valid:.4f}; ")

            if epoch_acc_rate_valid >= best_epoch_acc_rate_valid:
                best_epoch_acc_rate_valid = epoch_acc_rate_valid
                torch.save(self.model.state_dict(),
                           f"{self.target_path}/state_dict_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                           f"_{epoch_acc_rate_train:.4f}_{best_epoch_acc_rate_valid:.4f}.pth")

    def __do_train(self, epoch):
        self.model.train()

        epoch_loss_sum = 0
        epoch_acc_sum = 0

        ix = 0
        for label, texts in self.train_data_loader:
            label = label - 1
            label = label.to(self.device)

            encoded_batch = self.tokenizer.batch_encode_plus(
                texts,
                max_length=512,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            input_ids = encoded_batch['input_ids'].to(self.device)
            # attention_mask = encoded_batch['attention_mask'].to(self.device)

            with (torch.set_grad_enabled(True)):
                # output = self.model(input_ids, attention_mask)
                output = self.model(input_ids, None)
                loss = self.criterion(output, label)
                _, prediction = torch.max(output, 1)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                batch_acc = prediction.eq(label).sum()
                now_batch_size = label.shape[0]

                epoch_loss_sum += (loss * now_batch_size)
                epoch_acc_sum += batch_acc

                self.writer.add_scalars("batch_train", {"loss": loss}, self.train_batch_index)
                self.writer.add_scalars("batch_train", {"acc": batch_acc / now_batch_size}, self.train_batch_index)

                print('[%d/%d][%d/%d]\t%s\t loss: %.4f\t acc_rate: %.4f'
                      % (epoch, self.epochs_num, ix, self.train_data_size, datetime.now(), loss.item(), batch_acc / now_batch_size))

                self.train_batch_index += 1
                ix += self.batch_size

        epoch_loss = epoch_loss_sum / self.train_data_size
        epoch_acc_rate = epoch_acc_sum / self.train_data_size
        return epoch_loss, epoch_acc_rate

    def __do_valid(self, epoch):
        self.model.eval()

        epoch_loss_sum = 0
        epoch_acc_sum = 0

        ix = 0
        for label, texts in self.valid_data_loader:
            label = label - 1
            label = label.to(self.device)

            encoded_batch = self.tokenizer.batch_encode_plus(
                texts,
                max_length=512,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            input_ids = encoded_batch['input_ids'].to(self.device)
            # attention_mask = encoded_batch['attention_mask'].to(self.device)

            with (torch.set_grad_enabled(False)):
                # output = self.model(input_ids, attention_mask)
                output = self.model(input_ids, None)
                loss = self.criterion(output, label)
                _, prediction = torch.max(output, 1)
                batch_acc = prediction.eq(label).sum()
                now_batch_size = label.shape[0]
                epoch_loss_sum += (loss * now_batch_size)
                epoch_acc_sum += batch_acc

                self.writer.add_scalars("batch_valid", {"loss": loss}, self.valid_batch_index)
                self.writer.add_scalars("batch_valid", {"acc": batch_acc / now_batch_size}, self.valid_batch_index)

                print('[%d/%d][%d/%d]\t%s\t loss: %.4f'
                      % (epoch, self.epochs_num, ix, self.valid_data_size, datetime.now(), loss.item()))

                self.valid_batch_index += self.batch_size
                ix += 1

        epoch_loss = epoch_loss_sum / self.valid_data_size
        epoch_acc_rate = epoch_acc_sum / self.valid_data_size
        return epoch_loss, epoch_acc_rate


if __name__ == '__main__':
    Model = TextClassify(train_data_path="./resources", batch_size=32, epochs_num=2, lr=1e-4, target_path="./target")
    Model.train_model()