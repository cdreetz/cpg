## Part 1 Build GcP Detector

```Here we have a simple problem, given a DNA sequence (of N, A, C, G, T), count the number of CpGs in the sequence (consecutive CGs).

We have defined a few helper functions / parameters for performing this task.

We need you to build a LSTM model and train it to complish this task in PyTorch.

A good solution will be a model that can be trained, with high confidence in correctness.```

1. Start just by copying the provided code for seed, rand_seq and count_cpgs.  As well as the helpers for converting int and dna
2. Added an encoding method to use during prepare_data, which we do by setting np.zeros and then changing the 0 to a 1 based on the index of the character
3. Also apply torch.tensor since we will be using torch to train the model
4. Setting the hyperparameters took a few attempts, starting with lr 0.001 and epoch 10, 100 units, 1 layer and it seemed like i wasn't getting any improvements with different values, until i just set the epoch to 200 and let it go for a while, 100+ epoch at least
5. Using both torch utils TensorDataset and DataLoader to ensure consistency in the types
6. Model is really basic, just assigning our hyperparameter set to the values, with a single lstm and single linear layer
7. The original notebooks mentioned using a classifier and returning logits but in this case I'm not classifying sequences but rather predicting CP counts which is a continuous output
8. Again given the outputs we want use MSE for the loss and Adam as the optimizer.
9. I ran the code both on my mac and pc so the only thing you may need to change is cuda/mps for device