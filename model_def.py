import torch
import torch.nn as nn
from transformers import RobertaModel

class RobertaBiLSTM(nn.Module):
    def __init__(self):
        super(RobertaBiLSTM, self).__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.lstm = nn.LSTM(input_size=768, hidden_size=256, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(512, 2)  # 256 * 2 for bidirectional

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        lstm_out, _ = self.lstm(sequence_output)
        final_hidden = lstm_out[:, -1, :]
        logits = self.fc(final_hidden)
        return logits
