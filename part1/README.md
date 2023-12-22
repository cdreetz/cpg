## Part 1 Build GcP Detector

```
Here we have a simple problem, given a DNA sequence (of N, A, C, G, T), count the number of CpGs in the sequence (consecutive CGs).

We have defined a few helper functions / parameters for performing this task.

We need you to build a LSTM model and train it to complish this task in PyTorch.

A good solution will be a model that can be trained, with high confidence in correctness.
```

1. Start just by copying the provided code for seed, rand_seq and count_cpgs.  As well as the helpers for converting int and dna
2. Added an encoding method to use during prepare_data, which we do by setting np.zeros and then changing the 0 to a 1 based on the index of the character
3. Also apply torch.tensor since we will be using torch to train the model
4. Setting the hyperparameters took a few attempts, starting with lr 0.001 and epoch 10, 100 units, 1 layer and it seemed like i wasn't getting any improvements with different values, until i just set the epoch to 200 and let it go for a while, 100+ epoch at least
5. Using both torch utils TensorDataset and DataLoader to ensure consistency in the types
6. Model is really basic, just assigning our hyperparameter set to the values, with a single lstm and single linear layer
7. The original notebooks mentioned using a classifier and returning logits but in this case I'm not classifying sequences but rather predicting CP counts which is a continuous output
8. Again given the outputs we want use MSE for the loss and Adam as the optimizer.
9. I ran the code both on my mac and pc so the only thing you may need to change is cuda/mps for device


LSTM_HIDDEN: 100, LSTM_LAYER: 1, lr: 0.001
Epoch: [1/200], Training Loss: 17.2410, Validation Loss: 4.1815
Epoch: [2/200], Training Loss: 4.2879, Validation Loss: 4.1645
Epoch: [3/200], Training Loss: 4.2064, Validation Loss: 4.1591
Epoch: [4/200], Training Loss: 4.1979, Validation Loss: 4.1570
Epoch: [5/200], Training Loss: 4.1950, Validation Loss: 4.1612
Epoch: [6/200], Training Loss: 4.2047, Validation Loss: 4.1570
Epoch: [7/200], Training Loss: 4.1936, Validation Loss: 4.1598
Epoch: [8/200], Training Loss: 4.1920, Validation Loss: 4.1874
Epoch: [9/200], Training Loss: 4.2012, Validation Loss: 4.1567
Epoch: [10/200], Training Loss: 4.1916, Validation Loss: 4.1569
Epoch: ...
Epoch: [199/200], Training Loss: 0.0050, Validation Loss: 0.0075
Epoch: [200/200], Training Loss: 0.0060, Validation Loss: 0.0096
Mean Squared Error: 0.009587624781824
R2 Score: 0.9976935227172514