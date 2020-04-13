import sentencepiece as spm
from bert_serving.client import BertClient


class Bert():
    """ Bert model client
        Before usage, you need to run bert server.
    """

    def __init__(self, bert_model_path, client_ip='0.0.0.0'):
        self.bert_client = BertClient(ip=client_ip)
        self.spm_model = spm.SentencePieceProcessor()
        self.spm_model.load(bert_model_path + 'wiki-ja.model')

    def _parse(self, text):
        text = str(text).lower()
        encoded_texts = self.spm_model.EncodeAsPieces(text)
        encoded_texts = [t for t in encoded_texts if t.strip()]
        return encoded_texts

    def text2vec(self, texts):
        """

        Args:
            texts (list): 日本語文字列のリスト

        Returns:
            numpy array: テキストの分散表現テンソル
        """

        parsed_texts = list(map(self._parse, texts))
        tensor = self.bert_client.encode(parsed_texts, is_tokenized=True)
        return tensor
