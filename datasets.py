# ------------------------
# Tiny Shakespeare dataset
# ------------------------
import pathlib, urllib.request, re, collections
import torch

# A Tiny-Shakespeare speaker tag is its own line: "FIRST CITIZEN:" / "Gloucester:" /
# "DUKE VINCENTIO:". The first char is upper-case, then letters and spaces, then a
# trailing colon and nothing else on the line. 90.8% of the file is labeled dialogue
# so this pattern captures almost everything that matters.
_SPEAKER_RE = re.compile(r'^[A-Z][A-Za-z ]+:\s*$')

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

        # keep the raw text around for speaker-based sharding (non-IID)
        self.text = text
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


def speaker_shard_ids(text: str, stoi: dict, world_size: int):
    """Partition Tiny-Shakespeare text by speaker into `world_size` disjoint shards.

    Each speaker's dialogue is assigned to a single rank: speakers are sorted by
    total dialogue length descending, then round-robin'd across ranks so the heavy
    hitters spread evenly (rank 0 gets the 1st, 5th, 9th, ... largest speakers
    when W=4). Result: every rank sees a disjoint set of characters with roughly
    equal total token counts, while the across-rank distribution is genuinely
    non-IID -- different vocabulary, prosody, character voice per worker.

    Returns a list of length `world_size`: encoded id tensors, one per rank.
    """
    blocks = collections.defaultdict(list)
    cur = None
    for line in text.split("\n"):
        if _SPEAKER_RE.match(line):
            cur = line.rstrip(":").strip()
        elif cur is not None:
            blocks[cur].append(line)

    sp_text = {sp: "\n".join(lines) for sp, lines in blocks.items() if lines}
    by_len = sorted(sp_text.items(), key=lambda kv: -len(kv[1]))

    rank_chunks = [[] for _ in range(world_size)]
    for i, (_sp, t) in enumerate(by_len):
        rank_chunks[i % world_size].append(t)

    out = []
    for chunks in rank_chunks:
        joined = "\n".join(chunks)
        ids = torch.tensor([stoi[c] for c in joined if c in stoi], dtype=torch.long)
        out.append(ids)
    return out
