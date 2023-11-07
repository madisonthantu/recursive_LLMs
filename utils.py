def compute_coverage(doc_tok_ids, summ_tok_ids):
    """_summary_
    Args:
        doc_tok_ids (List[Int]): _description_
        summ_tok_ids (List[Int]): _description_
    REF: HydraSum: Disentangling Style Features in Text Summarization with Multi-Decoder Models
    What it do:
        Compute the average 'coverage' of a given dataset (document, summary pairs), where coverage denotes the fraction of summary words that are also present in the input (i.e., document)
    """
    pass


def compute_compression_ratio(doc_tok_ids, summ_tok_ids):
    """_summary_

    Args:
        doc_tok_ids (List[Int]): _description_
        summ_tok_ids (List[Int]): _description_
    REF: HydraSum: Disentangling Style Features in Text Summarization with Multi-Decoder Models
    What it do: A length metric; ratio of the number of words in the summary and the input document
    """
    pass


def compute_tok_distribution(summ_tok_ids):
    """_summary_

    Args:
        summ_tok_ids (List[Int]): _description_
    What it do: 
    """
    pass


def evaluate_emotion_intensity():
    pass


def evaluate_toxicity():
    pass


def evaluate_formality():
    pass


class Measurement:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer # *** Use NLTK ***
        self.generation = 0
        self.measurement_keys = ['coverage', 'compression_ratio', 'summary_token_distribution', 'document_vocab_size', 'summary_vocab_size', 'summary_samples']
        self.measurements = {}
        
    def measure(documents, summaries):
        pass