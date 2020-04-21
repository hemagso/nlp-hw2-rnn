from hw2.rnn import RNN
from hw2.data import DataPreprocessing
from hw2.utils import TimeKeeper

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd

data = DataPreprocessing("../data")

DEVICE = "cuda"

LR = 1E-4
HIDDEN_SIZE = 256

model = RNN(input_size=data.vocab_size, hidden_size=HIDDEN_SIZE, output_size=data.tag_size).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

def step(model, sentence, tag):
    hidden = model.init_hidden().to(DEVICE)
    outputs = torch.zeros(sentence.shape[0], data.tag_size, dtype=torch.float).to(DEVICE)
    for idx, word in enumerate(sentence):
        output, hidden = model(word, hidden)
        outputs[idx] = output

    loss = criterion(outputs, tag)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return outputs, loss.item()

N_EPOCHS = 5
PRINT_EVERY = 1000
tk = TimeKeeper()
iter_count = 0
all_losses = []
all_gradients = []


model.train()
for epoch_i in range(N_EPOCHS):
    for sentence, tag in data.train_iterator():
        sentence = sentence.to(DEVICE)
        tag = tag.to(DEVICE)
        output, loss = step(model, sentence, tag)

        if iter_count % PRINT_EVERY == 0:
            print("%d %s %.4f" % (iter_count, tk, loss))

        all_losses.append(loss)
        all_gradients.append(model.get_gradient_norm(2))

        iter_count += 1

stats = pd.DataFrame(all_losses, columns=["losses"])
gradients = pd.DataFrame(all_gradients, columns=model.get_parameter_names())

stats.to_csv("../models/rnn_01/stats.csv")
gradients.to_csv("../models/rnn_01/gradients.csv")
torch.save(model.state_dict(), "../models/rnn_01/weights.model")

def evaluate_result(true_tags, pred_tags):
    p_list = []
    r_list = []
    f1_list = []
    for true_tag, pred_tag in zip(true_tags, pred_tags):
        p, r, f1, _ = precision_recall_fscore_support(true_tag, pred_tag, average="macro", zero_division=0)
        p_list.append(p)
        r_list.append(r)
        f1_list.append(f1)
    return np.mean(p_list), np.mean(r_list), np.mean(f1_list)


def rnn_predict_one_sentence(model, sentence, device="cuda"):
    hidden = model.init_hidden().to(device)
    outputs = torch.zeros(sentence.shape[0], data.tag_size, dtype=torch.float).to(device)
    for idx, word in enumerate(sentence):
        outputs[idx], hidden = model(word, hidden)
    return outputs.argmax(axis=1)


model.eval()
predicted_tags = []
did = False
for sentence in data.test_data_oh:
    sentence = sentence.to(DEVICE)
    predicted_tag = rnn_predict_one_sentence(model, sentence, device=DEVICE)
    if not did:
        print(predicted_tag)
        did = True
    predicted_tags.append(predicted_tag.detach().cpu().numpy())

print(evaluate_result(data.test_tags_id, predicted_tags))

for name, p in model.named_parameters():
    print(name, p)
