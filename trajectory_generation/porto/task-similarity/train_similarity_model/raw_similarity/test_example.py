import torch
from torch.nn import Module, LSTM
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class MaskedLSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0., bidirectional=False):
        super(MaskedLSTM, self).__init__()
        self.batch_first = batch_first
        self.lstm = LSTM(input_size, hidden_size, num_layers=num_layers, bias=bias,
             batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)

    def forward(self, input_tensor, seq_lens):
        # input_tensor shape: batch_size*time_step*dim , seq_lens: (batch_size,)  when batch_first = True
        total_length = input_tensor.size(1) if self.batch_first else input_tensor.size(0)
        x_packed = pack_padded_sequence(input_tensor, seq_lens, batch_first=self.batch_first, enforce_sorted=False)
        y_lstm, hidden = self.lstm(x_packed)
        y_padded, length = pad_packed_sequence(y_lstm, batch_first=self.batch_first, total_length=total_length)
        return y_padded, hidden

rnn_example = MaskedLSTM(5, 5, batch_first=True)
input_tensor = torch.rand(6, 4, 5)
lens = [2, 2, 4, 3, 1, 3]
y_pad, hid = rnn_example(input_tensor, lens)
print(y_pad.shape)
print(hid[1].shape)
print(y_pad[5,2,:])
print(hid[0][:,5,:])