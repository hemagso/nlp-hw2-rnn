import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from hw2.rnn import RNN
from hw2.data import DataPreprocessing
import torch

data = DataPreprocessing("../data")
DEVICE = "cuda"

HIDDEN_SIZE = 256

model = RNN(input_size=data.vocab_size, hidden_size=HIDDEN_SIZE, output_size=data.tag_size)
model.load_state_dict(torch.load("../notebooks/test.model"))
model = model.to(DEVICE)

print(model.predict(data.train_data_oh[0].to(DEVICE)))

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

print(model.state_dict())

model.eval()
predicted_tags = []

#for sentence in data.test_data_oh:
#    sentence = sentence.to(DEVICE)
#    predicted_tag = model.predict(sentence, device=DEVICE).argmax(axis=1)
#    predicted_tags.append(predicted_tag.detach().cpu().numpy())

# print(evaluate_result(data.test_tags_id, predicted_tags))

