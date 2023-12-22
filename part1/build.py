from typing import Sequence
from functools import partial
import random
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import random

def set_seed(seed=13):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(13)

def rand_sequence(n_seqs: int, seq_len: int=128) -> Sequence[int]:
    for i in range(n_seqs):
        yield [random.randint(0, 4) for _ in range(seq_len)]

def count_cpgs(seq: str) -> int:
    cgs = 0
    for i in range(0, len(seq) - 1):
        dimer = seq[i:i+2]
        if dimer == "CG":
            cgs += 1
    return cgs

alphabet = 'NACGT'
dna2int = { a: i for a, i in zip(alphabet, range(5))}
int2dna = { i: a for a, i in zip(alphabet, range(5))}

intseq_to_dnaseq = partial(map, int2dna.get)
dnaseq_to_intseq = partial(map, dna2int.get)

def encode(sequence, num_classes=5):
    one_hot = np.zeros((len(sequence), num_classes), dtype=np.float32)
    for idx, val in enumerate(sequence):
        one_hot[idx, val] = 1.0

    return one_hot

def prepare_data(num_samples=100):
    X_dna_seqs_train = list(rand_sequence(num_samples))
    X_dna_seqs_train_one_hot = [encode(seq) for seq in X_dna_seqs_train]

    X_dna_seqs_train_one_hot_np = np.array(X_dna_seqs_train_one_hot)

    temp = [''.join(intseq_to_dnaseq(seq)) for seq in X_dna_seqs_train]
    y_dna_seqs = [count_cpgs(seq) for seq in temp]

    X_dna_seqs_train_tensor = torch.tensor(X_dna_seqs_train_one_hot_np, dtype=torch.float)
    y_dna_seqs_tensor = torch.tensor(y_dna_seqs, dtype=torch.float)

    return X_dna_seqs_train_tensor, y_dna_seqs_tensor


train_x, train_y = prepare_data(2048)
test_x, test_y = prepare_data(512)

LSTM_HIDDEN = 100
LSTM_LAYER = 1
batch_size = 64
learning_rate = 0.001
epoch_num = 200



train_dataset = TensorDataset(train_x, train_y)
test_dataset = TensorDataset(test_x, test_y)

train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class CpGPredictor(torch.nn.Module):
    def __init__(self):
        super(CpGPredictor, self).__init__()

        self.lstm = nn.LSTM(input_size=5,
                            hidden_size=LSTM_HIDDEN,
                            num_layers=LSTM_LAYER,
                            batch_first=True)

        self.classifier = nn.Linear(in_features=LSTM_HIDDEN,
                                    out_features=1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        logits = self.classifier(lstm_out)

        return logits

model = CpGPredictor()
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

print(f"LSTM_HIDDEN: {LSTM_HIDDEN}, LSTM_LAYER: {LSTM_LAYER}, lr: {learning_rate}")

for epoch in range(epoch_num):
    model.train()
    t_loss = 0.0
    for batch in train_data_loader:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        outputs = outputs.squeeze()
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        t_loss += loss.item()

    avg_train_loss = t_loss / len(train_data_loader)

    model.eval()
    v_loss = 0.0
    with torch.no_grad():
        for batch in test_data_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            outputs = outputs.squeeze()

            loss = loss_fn(outputs, labels)
            v_loss += loss.item()

    avg_val_loss = v_loss / len(test_data_loader)

    print(f"Epoch: [{epoch+1}/{epoch_num}], Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

res_gs = []
res_pred = []

with torch.no_grad():
    for i, batch in enumerate(test_data_loader):
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)

        actual_labels = labels.tolist()
        predicted_labels = outputs.tolist()

        batch_mse = mean_squared_error(actual_labels, predicted_labels)

        #print(f"Batch {i} MSE: {batch_mse}")

        res_gs.extend(labels.tolist())
        res_pred.extend(outputs.tolist())


        #print(f"Processed {i}/{len(test_data_loader)} batches.")

mse = mean_squared_error(res_gs, res_pred)
print(f"Mean Squared Error: {mse}")

r2 = r2_score(res_gs, res_pred)
print(f"R2 Score: {r2}")


