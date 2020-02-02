import torch.nn as nn

class language_model(nn.Module):
    def __init__(self, num_words, embed_dim):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings=num_words, embedding_dim=embed_dim) #size: (*) -> (*,embedding_dim)
        self.dropout1 = nn.Dropout(p=0.3)
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=embed_dim, num_layers=2)
        self.linear = nn.Linear(in_features=embed_dim, out_features=num_words)


    def forward(self, x):
        '''
        :param x: tensor of indices of words, size (seq_len, batch_size)
        :return: tensor of probabilities for words at each time step
        '''
        output = self.embed(x)
        output = self.dropout1(output)
        output, _ = self.lstm(output) #size: (seq_len, batch_size, hidden_size)
        output = self.linear(output) #size: (seq_len, batch_size, num_words)
        return output
