Part 2 Sequences of Different Length

build3.py is the final version

In sequence data like the data we are working with here, it isn't always that all sequences are of the same length.
Per the data creation step here, sequences can range from a length of 16 to 128.

Padding is used on sequences of shorter length, to fill the remainder of the sequence until all seqs are the same length
Packing is used as a form of hierarchial references to seqs of different lengths.
The goal of packing is to save on unnecessary computations by removing the need to mul padded or 0 values

1. Import pad_sequence, pack_padded_sequence, pad_packed_sequence
2. Copy the data creating steps that should not be changed
3. Similar steps to the first part of the assignment, but this time in our data prep we apply pad_sequence
4. Then in the model step itself, we use pack_padded_sequence to order sequences by their actual lengths 
5. After lstm is done we convert the packed back to regular padded sequences
6. Same thing here you may need to change cuda to mps or vice versa
7. One note is that in the case of using GPU you have to ensure that inputs and labels are put on GPU but lengths stay on CPU

```
LSTM_HIDDEN: 50, LSTM_LAYER: 1, lr: 0.001
Epoch: [1/200], Training Loss: 15.892235338687897, Val Loss: 14.702741384506226
Epoch: [2/200], Training Loss: 10.397656410932541, Val Loss: 5.579729676246643
Epoch: [3/200], Training Loss: 4.071665719151497, Val Loss: 3.9627086520195007
Epoch: [4/200], Training Loss: 3.7221716940402985, Val Loss: 3.9358795881271362
Epoch: [5/200], Training Loss: 3.647277921438217, Val Loss: 3.919661283493042
Epoch: [6/200], Training Loss: 3.6462442129850388, Val Loss: 3.911962389945984
Epoch: [7/200], Training Loss: 3.6418788135051727, Val Loss: 3.9114903807640076
Epoch: [8/200], Training Loss: 3.641308143734932, Val Loss: 3.9108508229255676
Epoch: [9/200], Training Loss: 3.641021654009819, Val Loss: 3.910228967666626
Epoch: [10/200], Training Loss: 3.640531465411186, Val Loss: 3.909626841545105
Epoch: [11/200], Training Loss: 3.6401010006666183, Val Loss: 3.909007966518402
Epoch: ...
Epoch: [195/200], Training Loss: 0.026041646138764918, Val Loss: 0.025809308979660273
Epoch: [196/200], Training Loss: 0.02365886908955872, Val Loss: 0.024485605768859386
Epoch: [197/200], Training Loss: 0.02376479998929426, Val Loss: 0.025473705492913723
Epoch: [198/200], Training Loss: 0.027059815009124577, Val Loss: 0.024679989088326693
Epoch: [199/200], Training Loss: 0.027024175797123462, Val Loss: 0.023339663166552782
Epoch: [200/200], Training Loss: 0.025369752023834735, Val Loss: 0.02368619292974472
Final Mean Squared Error: 0.0236861924572005
Final R² Score: 0.9939486206856388
```