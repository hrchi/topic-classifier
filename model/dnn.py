import torch.nn as nn
import torch.nn.functional as F


class DNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout_rate = dropout

        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        self.fc3 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        pooled = embedded.mean(dim=1)
        
        x = self.fc1(pooled)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout_rate, training=self.training)
        
        #x = self.fc2(x)
        #x = self.bn2(x)
        #x = F.relu(x)
        #x = F.dropout(x, self.dropout_rate, training=self.training)
        
        x = self.fc3(x)
        return x
