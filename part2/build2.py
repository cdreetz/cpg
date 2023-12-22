from typing import Sequence
from functools import partial
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import random

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


LSTM_HIDDEN = 20
LSTM_LAYER = 2
batch_size = 128
learning_rate = 0.001
epoch_num = 200

random.seed(13)

def rand_sequence_var_len(n_seqs: int, lb: int=16, ub: int=128) -> Sequence[int]:
    for i in range(n_seqs):
        seq_len = random.randint(lb, ub)
        yield [random.randint(1, 5) for _ in range(seq_len)]

def count_cpgs(seq: str) -> int:
    cgs = 0
    for i in range(0, len(seq) - 1):
        dimer = seq[i:i+2]
        if dimer == "CG":
            cgs += 1
    return cgs

alphabet = 'NACGT'
dna2int = {a: i for a, i in zip(alphabet, range(1, 6))}
int2dna = {i: a for a, i in zip(alphabet, range(1, 6))}
dna2int.update({"pad": 0})
int2dna.update({0: "<pad>"})

intseq_to_dnaseq = partial(map, int2dna.get)
dnaseq_to_intseq = partial(map, dna2int.get)

def prepare_data(num_samples=100, min_len=16, max_len=128):
    X_dna_seqs_train = list(rand_sequence_var_len(num_samples, min_len, max_len))
    temp = [''.join(intseq_to_dnaseq(seq)) for seq in X_dna_seqs_train]
    y_dna_seqs = [count_cpgs(seq) for seq in temp]
    X_dna_seqs_train_tensors = [torch.tensor(seq, dtype=torch.long) for seq in X_dna_seqs_train]
    X_dna_seqs_train_padded = pad_sequence(X_dna_seqs_train_tensors, batch_first=True, padding_value=dna2int['pad'])

    return X_dna_seqs_train_padded, y_dna_seqs

min_len, max_len = 64, 128
train_x, train_y = prepare_data(2048, min_len, max_len)
test_x, test_y = prepare_data(512, min_len, max_len)


def one_hot_encode(sequences, sequence_length, num_classes=6):
    one_hot = torch.zeros((len(sequences), sequence_length, num_classes), dtype=torch.float32)
    for i, sequence in enumerate(sequences):
        for j, item in enumerate(sequence):
            one_hot[i, j] = 1

    return one_hot

train_x = one_hot_encode(train_x, max_len)
test_x = one_hot_encode(test_x, max_len)


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, lists, labels) -> None:
        self.lists = lists
        self.labels = labels

    def __getitem__(self, index):
        return self.lists[index], self.labels[index]

    def __len__(self):
        return len(self.lists)



class PadSequence:
    def __call__(self, batch):
        sequences, labels = zip(*batch)
        lengths = torch.tensor([len(seq) for seq in sequences])
        padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=dna2int['pad'])
        labels = torch.tensor(labels)

        return padded_sequences, lengths, labels


train_dataset = MyDataset(train_x, train_y)
test_dataset = MyDataset(test_x, test_y)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, collate_fn=PadSequence())
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, collate_fn=PadSequence())


class CpGPredictor(torch.nn.Module):
    def __init__(self):
        super(CpGPredictor, self).__init__()

        self.lstm = nn.LSTM(input_size=6,
                            hidden_size=LSTM_HIDDEN,
                            num_layers=LSTM_LAYER,
                            batch_first=True)

        self.classifier = nn.Linear(in_features=LSTM_HIDDEN,
                                    out_features=1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        logits = self.classifier(lstm_out)

        return torch.relu(logits)

model = CpGPredictor()
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

print(f"LSTM_HIDDEN: {LSTM_HIDDEN}, LSTM_LAYER: {LSTM_LAYER}, lr: {learning_rate}")

for epoch in range(epoch_num):
    model.train()
    t_loss = 0.0
    for batch in train_loader:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        outputs = outputs.squeeze()
        loss = loss_fn(outputs, labels.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        t_loss += loss.item()

    avg_train_loss = t_loss / len(train_loader)

    model.eval()
    v_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            outputs = outputs.squeeze()

            loss = loss_fn(outputs, labels.float())
            v_loss += loss.item()

    avg_val_loss = v_loss / len(test_loader)

    print(f"Epoch: [{epoch+1}/{epoch_num}], Training Loss: {avg_train_loss}, Val Loss: {avg_val_loss}")


model.eval()
res_gs = []
res_pred = []  # List to store model predictions

with torch.no_grad():
    for batch in test_loader:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        outputs = outputs.squeeze()

        res_gs.extend(labels.tolist())
        res_pred.extend(outputs.tolist())

# Calculate and print final MSE and R² Score
final_mse = mean_squared_error(res_gs, res_pred)
final_r2 = r2_score(res_gs, res_pred)
print(f"Final Mean Squared Error: {final_mse}")
print(f"Final R² Score: {final_r2}")
