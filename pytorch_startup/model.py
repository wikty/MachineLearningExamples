import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Net(nn.Module):
    """An recurrent network that predicts the NER tags for each token in the
    sentence. The components of network as following:

    * an word-embedding layer
    * a LSTM recurrent layer
    * a fully connected layer
    """

    def __init__(self, word_vocab_size, embedding_dim, 
                 lstm_hidden_dim, tag_vocab_size):
        super().__init__()
        # Embedding Layer:
        # map a index of word to a word-embedding vector
        self.embedding = nn.Embedding(word_vocab_size, embedding_dim)
        # LSTM Layer:
        # map a sequence of embedding vectors to a sequence of hidden vectors
        self.lstm = nn.LSTM(embedding_dim, lstm_hidden_dim, batch_first=True)
        # Full Connection Layer:
        # map all of hidden vectors to the final output layer
        self.fc = nn.Linear(lstm_hidden_dim, tag_vocab_size)

    def forward(self, x):
        """Forward mini-batch in network.
        
        Args:
            x (Tensor): is a mini-batch of sentences. Shape is (batch_size, 
                seq_len). seq_len is the length of the longest sentence in 
                the batch. For sentences shorter than seq_len, will be 
                appended with word padding. Each row is a sentence with each
                element corresponding to the index of the token in the vocab. 
        
        Returns:
            Tensor: The result is a tensor with (batch_size*seq_len, 
            tag_vocab_size) shape. Each row contains the log probabilities of
            tags for a token.
        """
        # embeddings: (batch_size, seq_len, embedding_dim)
        embeddings = self.embedding(x)
        
        # outputs: (batch_size, seq_len, lstm_hidden_dim)
        outputs, hidden = self.lstm(embeddings)

        # call contiguous() before view()
        outputs = outputs.contiguous()

        # outputs: (batch_size*seq_len, lstm_hidden_dim)
        outputs = outputs.view(-1, outputs.size()[-1])

        # scores: (batch_size*seq_len, tag_vocab_size)
        scores = self.fc(outputs)

        # log_probs: (batch_size*seq_len, tag_vocab_size)
        log_probs = F.log_softmax(scores, dim=1)

        return log_probs


def criterion(outputs, targets):
    """Compute the cross entropy loss given outputs from the model and 
    targets/labels for all tokens. Exclude loss for padding tokens.

    Args:
        outputs (Tensor): The shape is (batch_size*seq_len, tag_vocab_size).
            Each row is the log probabilities of tags for a token.
        targets (Tensor): The shape is (batch_size, seq_len). Each row is
            a sentence with element corresponding to the tag index for each 
            token in this sentence.

    Returns:
        Tensor: the scalar tensor of the loss of mini-batch.

    Note: loss computation should be tracked by auto-diff.
    """
    batch_size, seq_len = targets.size()
    # flatten and get indices for non-padding tokens
    targets = targets.view(-1)  # (batch_size*seq_len, )
    indices = (targets >= 0).float()
    # negative index would be a problem, so assign to a positive value
    targets[targets < 0] = 1
    # discard the loss for padding tokens
    log_probs = outputs[range(batch_size*seq_len), targets] * indices
    return -torch.sum(log_probs) / torch.sum(indices)


def accuracy(outputs, targets):
    """Calculate the accuracy of mini-batch.

    Args:
        outputs (LongTensor): The shape is (batch_size*seq_len, 
            tag_vocab_size). Each row is the log probabilities of tags for a 
            token.
        targets (LongTensor): The shape is (batch_size, seq_len). Each row is
            a sentence with element corresponding to the tag index for each 
            token in this sentence.

    Returns:
        Tensor: return the scalar tensor of prediction accuracy.

    Note: Accuracy excludes the padding tags. The padding tag in `targets`
    is `-1`.
    """
    # # flatten
    # targets = targets.ravel()  # (batch_size*seq_len, )
    # indices = (targets >= 0)
    # # calculate the predictions
    # predictions = np.argmax(outputs, axis=1)  # (batch_size*seq_len, )
    # return np.sum((targets == predictions)[indices]) / float(np.sum(indices))
    with torch.no_grad():
        targets = targets.view(-1)  # flatten: (batch_size*seq_len, )
        indices = (targets >= 0)
        predictions = torch.argmax(outputs, dim=1)  # (batch_size*seq_len, )
        # must convert to float
        return (torch.sum((targets == predictions) * indices).float() /
            torch.sum(indices).float())


# maintain all metrics required in this dictionary- these are used in the
# training and evaluation loops.
metrics = {
    'accuracy': accuracy,
    # could add more metrics such as accuracy for each token type
}

def model_factory(params):
    model = Net(params.word_vocab_size, 
                params.embedding_dim, 
                params.lstm_hidden_dim, 
                params.tag_vocab_size)
    if params.cuda:
        model = model.cuda()
    optimizer = optim.Adam(model.parameters(), 
                           lr=params.learning_rate)
    return model, optimizer, criterion, metrics