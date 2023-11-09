from googleapiclient import discovery
import json
import numpy as np

def init():
    # Admin globals
    global new_col_names, keep_cols, rng
    new_col_names = {
        'article':'document',
        'highlights':'summary',
        'documents':'document',
        'tldr':'summary',
        'dialogue':'document'
    }
    keep_cols = ['document', 'summary', 'id']
    rng = np.random.default_rng()
    # Globals for toxicity eval
    global API_KEY, client
    API_KEY = 'AIzaSyDcA-LYHVNateEydAvPLg5AaF19sZwM-mY'
    client = discovery.build(
        "commentanalyzer",
        "v1alpha1",
        developerKey=API_KEY,
        discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
        static_discovery=False,
    )
    # Globals for formality eval
    global API_URL, headers
    API_URL = "https://api-inference.huggingface.co/models/s-nlp/roberta-base-formality-ranker"
    headers = {"Authorization": "Bearer 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"}
