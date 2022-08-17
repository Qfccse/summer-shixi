import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

dtype = torch.FloatTensor
char_list = [c for c in 'SEPabcdefghijklmnopqrstuvwxyz']
char_dic = {n: i for i, n in enumerate(char_list)}
print(char_list)
print(char_dic)
seq_data = [['man', 'women'], ['black', 'white'], ['king', 'queen'], ['girl', 'boy'], ['up', 'down'], ['high', 'low']]
seq_len = 8
n_hidden = 128
n_class = len(char_list)
batch_size = len(seq_data)


def make_batch(seq_data):
    batch_size = len(seq_data)
    input_batch, output_batch, target_batch = [], [], []
    for seq in seq_data:
        for i in range(2):
            seq[i] += 'P' * (seq_len - len(seq[i]))
        input = [char_dic[n] for n in seq[0]]
        output = [char_dic[n] for n in ('S' + seq[1])]
        target = [char_dic[n] for n in (seq[1] + 'E')]

        input_batch.append(np.eye(n_class)[input])
        output_batch.append(np.eye(n_class)[output])
        target_batch.append(target)

    return Variable(torch.Tensor(input_batch)), Variable(torch.Tensor(output_batch)), Variable(
        torch.LongTensor(target_batch))


input_batch, output_batch, target_batch = make_batch(seq_data)


class Seq2Seq(nn.Module):
    def __init__(self):
        super(Seq2Seq, self).__init__()
        self.encoder = nn.RNN(input_size=n_class, hidden_size=n_hidden)
        self.decoder = nn.RNN(input_size=n_class, hidden_size=n_hidden)
        self.fc = nn.Linear(n_hidden, n_class)

    def forward(self, enc_input, enc_hidden, dec_input):
        enc_input = enc_input.transpose(0, 1)
        dec_input = dec_input.transpose(0, 1)

        _, h_states = self.encoder(enc_input, enc_hidden)
        outputs, _ = self.decoder(dec_input, h_states)
        outputs = self.fc(outputs)
        return outputs


model = Seq2Seq()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5001):
    hidden = Variable(torch.zeros(1, batch_size, n_hidden))
    optimizer.zero_grad()

    outputs = model(input_batch, hidden, output_batch)
    outputs = outputs.transpose(0, 1)

    loss = 0
    for i in range(batch_size):
        loss += criterion(outputs[i], target_batch[i])
    if (epoch % 500) == 0:
        print('epoch:{},loss:{}'.format(epoch, loss))
    loss.backward()
    optimizer.step()


def translate(word):
    input_batch, output_batch, _ = make_batch([[word, 'P' * len(word)]])
    hidden = Variable(torch.zeros(1, 1, n_hidden))
    outputs = model(input_batch, hidden, output_batch)
    predict = outputs.data.max(2, keepdim=True)[1]
    decode = [char_list[i] for i in predict]
    end = decode.index('P')
    translated = ''.join(decode[:end])
   # print(translated)

    return translated

print('test')
print('man ->', translate('man'))
print('mans ->', translate('mans'))
print('king ->', translate('king'))
print('black ->', translate('black'))
print('up ->', translate('up'))
