import torch.nn as nn
import torch


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.W_hi = LSTM.create_parameter(hidden_size, hidden_size)
        self.W_hf = LSTM.create_parameter(hidden_size, hidden_size)
        self.W_ho = LSTM.create_parameter(hidden_size, hidden_size)
        self.W_hh = LSTM.create_parameter(hidden_size, hidden_size)

        self.W_xi = LSTM.create_parameter(hidden_size, input_size)
        self.W_xf = LSTM.create_parameter(hidden_size, input_size)
        self.W_xo = LSTM.create_parameter(hidden_size, input_size)
        self.W_xh = LSTM.create_parameter(hidden_size, input_size)

        self.b_i = LSTM.create_parameter(hidden_size, 1)
        self.b_f = LSTM.create_parameter(hidden_size, 1)
        self.b_o = LSTM.create_parameter(hidden_size, 1)
        self.b_h = LSTM.create_parameter(hidden_size, 1)

        self.W_hy = LSTM.create_parameter(output_size, hidden_size)
        self.b_y = LSTM.create_parameter(output_size, 1)

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

    def forward(self, input, hidden, memory):
        i = torch.sigmoid(self.W_hi @ hidden.T + self.W_xi @ input.view(-1, 1) + self.b_i).T
        f = torch.sigmoid(self.W_hf @ hidden.T + self.W_xf @ input.view(-1, 1) + self.b_f).T
        o = torch.sigmoid(self.W_ho @ hidden.T + self.W_xo @ input.view(-1, 1) + self.b_o).T

        memory_ = torch.tanh(self.W_hh @ hidden.T + self.W_xh @ input.view(-1, 1) + self.b_h).T
        memory = f * memory + i * memory_

        hidden = o * torch.tanh(memory)
        output = (self.W_hy @ hidden.T + self.b_y).T

        return output, hidden, memory

    def init_hidden(self):
        return (torch.zeros(1, self.hidden_size),
                torch.zeros(1, self.hidden_size))

    def predict(self, sentence, device="cuda"):
        hidden, memory = self.init_hidden()
        hidden = hidden.to(device)
        memory = memory.to(device)
        outputs = torch.zeros(sentence.shape[0], self.output_size, dtype=torch.float).to(device)
        for idx, word in enumerate(sentence):
            outputs[idx], hidden, memory = self(word, hidden, memory)
        return outputs
