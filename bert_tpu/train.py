import transformers
from torch import nn


class BertBaseUncased(nn.Module):

    def __init__(self, bert_path):
        super(BertBaseUncased, self).__init__()
        self.bert_path = bert_path
        self.bert = transformers.BertModel.from_pretrained(self.bert_path)
        self.dropout = nn.Dropout(0.3)
