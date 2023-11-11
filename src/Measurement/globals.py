from googleapiclient import discovery
import json
import numpy as np

def init():
    # Housekeeping globals
    global new_col_names, keep_cols, rng, max_src_length, max_target_length, sample_idxs, dataset_files
    
    new_col_names = {
        'article':'document',
        'highlights':'summary',
        'documents':'document',
        'tldr':'summary',
        'dialogue':'document'
    }
    max_src_length = {
        'news':1024,
        'dialogue':512,
        'reddit':1024,
    }
    max_target_length = {
        'news':256,
        'dialogue':75,
        'reddit':75,
    }
    dataset_files = {
        'reddit': 'reddit_initial_dataset.pkl',
        'dialogue': 'dialogue_initial_dataset.pkl',
        'news': 'news_initial_dataset.pkl'
    }
    keep_cols = ['document', 'summary', 'id']
    rng = np.random.default_rng()
    # sample_idxs = np.array([12533, 13142, 9985, 6799, 3970, 7474, 7520, 11167, 4330, 357, 16035, 11539, 12455, 475, 8286, 5405, 7215, 15558, 2597, 6688, 10471, 13918, 9739, 7910, 1864, 10679, 3770, 13883, 12911, 9004, 13607, 11250, 5718, 10430, 10488, 961, 12333, 2321, 14797, 12491, 7327, 16221])
    sample_idxs = [i for i in range(0,10)]
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
    # Globals for HuggingFace
    global hug_token
    hug_token = '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'
    
    # MISC ...
    global emoji_dict
    emoji_dict = {
        0: "⁺✧.(˃̶ ॣ⌣ ॣ˂̶∗̀)ɞ⁾",
        1: "ଘ(੭ˊ◕ᵕ◕ˋ)੭†",
        2: "乁༼☯‿☯✿༽ㄏ",
        3: "─=≡Σ((( つ•̀ω•́)つ",
        4: "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀\n⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣴⠶⢦⣤⠶⠶⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀\n⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣇⠀⠀⠁⠀⢀⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀\n⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢧⣄⠀⣠⠞⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀\n⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⡀⠀⠉⠛⠃⣠⣄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀\n⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡞⠉⠙⢳⣄⢀⡾⠁⠈⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀\n⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢻⡄⠀⠀⠙⢿⡇⠀⢰⠇⠀⠀⠀⠀⠀⠀⠀⠀⠀\n⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⣦⡀⠀⠀⠹⣦⡟⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀\n⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⢳⣄⠀⠀⠈⠻⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀\n⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡞⠋⠛⢧⡀⠀⠀⠘⢷⡀⠀⠀⠀⠀⠀⠀⠀\n⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⡴⠾⣧⡀⠀⠀⠹⣦⠀⠀⠈⢿⡄⠀⠀⠀⠀⠀⠀\n⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣿⠀⠀⠈⠻⣄⠀⠀⠀⠀⠀⠀⠈⣷⠀⠀⠀⠀⠀⠀\n⠀⠀⠀⠀⠀⠀⠀⢠⡟⠉⠛⢷⣄⠀⠀⠈⠀⠀⠀⠀⠀⠀⣰⠏⠀⠀⠀⠀⠀⠀\n⠀⠀⠀⠀⠀⠀⠀⠀⢷⡀⠀⠀⠉⠃⠀⠀⠀⠀⠀⠀⠀⣴⠏⠀⠀⠀⠀⠀⠀⠀\n⠀⠀⠀⠀⠀⠀⠀⠀⠈⠻⣦⡀⠀⠀⠀⠀⠀⠀⢀⣠⠞⠁⠀⠀⠀⠀⠀⠀⠀⠀\n⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠙⠶⣤⣤⣤⡤⠶⠋⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀\n⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀\n⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀",
            5: "(✿´ ‿`)🍨🍨(╹ワ╹✿)",
    }