import torch.nn as nn
import torch


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.W_hy = RNN.create_parameter(output_size, hidden_size)
        self.W_hh = RNN.create_parameter(hidden_size, hidden_size)
        self.b_h = RNN.create_parameter(hidden_size, 1)
        self.W_xh = RNN.create_parameter(hidden_size, input_size)
        self.b_y = RNN.create_parameter(output_size, 1)

    def get_gradient_norm(self, norm=2):
        return [p.grad.data.norm(norm).item() for p in self.parameters()]

    def get_parameter_names(self):
        return [name for name, _ in self.named_parameters()]

    @staticmethod
    def create_parameter(*size):
        parameter = nn.Parameter(torch.FloatTensor(*size))
        parameter.requires_grad = True
        nn.init.xavier_uniform_(parameter)
        return parameter

    def forward(self, input, hidden):
        # (HIDDEN, HIDDEN) x (HIDDEN, 1) + (HIDDEN, INPUT) x (INPUT, 1) + (HIDDEN, 1)
        hidden = nn.functional.relu(self.W_hh @ hidden.T + self.W_xh @ input.view(-1, 1) + self.b_h).T
        # (OUTPUT, HIDDEN) x (HIDDEN, 1) + (OUTPUT, 1) = (OUTPUT, 1)
        output = (self.W_hy @ hidden.T + self.b_y).T
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size, dtype=torch.float)

    def predict(self, sentence, device="cuda"):
        hidden = self.init_hidden().to(device)
        outputs = torch.zeros(sentence.shape[0], self.output_size, dtype=torch.float).to(device)
        for idx, word in enumerate(sentence):
            outputs[idx], hidden = self(word, hidden)
        return outputs
