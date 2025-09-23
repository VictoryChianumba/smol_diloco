# ------------------------
# Tiny Shakespeare dataset
# ------------------------
import pathlib, urllib.request
import torch

# ------------------------
# Tiny toy tokenizer/data
# ------------------------
class ToyCharDataset(torch.utils.data.Dataset):
    def __init__ (self, length: int=200000, ctx:int=64, vocab:int=96, seed:int=123):
        g = torch.Generator().manual_seed(seed) 
        self.data = torch.randint(low=0, high=vocab, size = (length+1,),generator = g)
        # What does this stand for?
        self.ctx = ctx
    
    def __len__(self):
        return len(self.data) - self.ctx - 1
    
    def __getitem__(self, idx):
        x = self.data[idx:idx+self.ctx]
        y = self.data[idx+1:idx+self.ctx+1]
        return x, y

  

class TinyShakespeareDataset(torch.utils.data.Dataset):
    def __init__(self, ctx: int = 128, path: str = "tiny_shakespeare.txt"):
        self.ctx = ctx
        path = pathlib.Path(path)
        if not path.exists():
            url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
            print(f"Downloading Tiny Shakespeare to {path} ...")
            urllib.request.urlretrieve(url, path)

        text = path.read_text(encoding="utf-8")
        # deterministic char vocab
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(chars)

        # encode to tensor
        self.data  = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)

    def __len__(self):
        return len(self.data) - self.ctx - 1

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.ctx]
        y = self.data[idx + 1 : idx + self.ctx + 1]
        return x, y

# A lightweight wrapper for a pre-sliced 1D id tensor (train/val splits)
class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, ids: torch.Tensor, ctx: int):
        self.ids = ids
        self.ctx = ctx
    def __len__(self):
        return len(self.ids) - self.ctx - 1
    def __getitem__(self, idx):
        x = self.ids[idx : idx + self.ctx]
        y = self.ids[idx + 1 : idx + self.ctx + 1]
        return x, y
