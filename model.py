import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import torch.autograd as autograd

class BowModel(nn.Module):
    def __init__(self, emb_tensor):
        super(BowModel, self).__init__()
        n_embedding, dim = emb_tensor.size()
        self.embedding = nn.Embedding(n_embedding, dim, padding_idx=0)
        self.embedding.weight = Parameter(emb_tensor, requires_grad=False)
        self.out = nn.Linear(dim, 2)

    def forward(self, input):
        '''
        input is a [batch_size, sentence_length] tensor with a list of token IDs
        '''
        embedded = self.embedding(input)
        # Here we take into account only the first word of the sentence
        # You should change it, e.g. by taking the average of the words of the sentence
        bow = embedded.mean(dim=1)
        return F.log_softmax(self.out(bow))



class BowModel(nn.Module):
	def __init__(self, emb_tensor):
		super(BowModel, self).__init__()
		
		n_embedding, dim = emb_tensor.size()
		self.embedding = nn.Embedding(n_embedding, dim, padding_idx=0)
		self.embedding.weight = Parameter(emb_tensor, requires_grad=False)
		
		self.hidden_dim = 6
		self.lstm = nn.LSTM(dim, self.hidden_dim)

		self.hidden2sentiment = nn.Linear(self.hidden_dim, 2)
		self.hidden = self.init_hidden()

	def init_hidden(self):
		return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
				autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

	def forward(self, input):
		self.hidden = self.init_hidden()
		embeds = self.embedding(input)
		emb = embeds.transpose(0, 1)
		lstm_out, self.hidden = self.lstm(emb, self.hidden)
		tag_space = self.hidden2sentiment(lstm_out[-1])
		tag_scores = F.log_softmax(tag_space)

		return tag_scores
