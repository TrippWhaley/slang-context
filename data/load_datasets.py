from datasets import load_dataset

def load_slang_dataset():
    slang_dataset = load_dataset("MLBtrio/genz-slang-dataset").data['train'].to_pandas()
    return slang_dataset

def load_reddit_teenagers_dataset():
    reddit_comments = load_dataset("amitrajitbh1/reddit_teen").data['train'].to_pandas()
    return reddit_comments
