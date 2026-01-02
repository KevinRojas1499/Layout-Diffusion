import torch
from torch.utils.data import Dataset
from utils.tokenizer import CharacterTokenizer
            
class ShakespeareDataset(Dataset):
    def __init__(self, context_len=30, data_path='shakespeare.txt'):
        self.context_len = context_len
        self.data_path = data_path
        # Open the data file and read the text
        characters = set() 
        with open(data_path, 'r') as file:
            self.text = file.read()
            for char in self.text:
                if char not in characters:
                    characters.add(char)
        
        self.char_to_idx = {char: idx for idx, char in enumerate(characters)}
        self.idx_to_char = {idx: char for idx, char in enumerate(characters)}
        # Add a padding token and a mask token
        self.char_to_idx['<pad>'] = len(characters)
        self.char_to_idx['<M>'] = len(characters) + 1
        self.idx_to_char[len(characters)] = '<pad>'
        self.idx_to_char[len(characters) + 1] = '<M>'
        self.vocab_size = len(characters) + 2
        self.pad_token = self.char_to_idx['<pad>']
        self.mask_token = self.char_to_idx['<M>']
        print(f'Initializing ShakespeareDataset with {len(self.char_to_idx)} characters')

    def __getitem__(self, index):
        chunk = self.text[index:index+self.context_len]
        # Take a random smaller chunk and add padding tokens to the end
        chunk = torch.tensor([self.char_to_idx[char] for char in chunk], dtype=torch.long)

        chunk = chunk[:torch.randint(self.context_len//2, len(chunk), (1,))]
        chunk = torch.cat((chunk, torch.ones(self.context_len - len(chunk), dtype=torch.long) * self.pad_token))
        return chunk
    def __len__(self):
        return len(self.text) - self.context_len

class EuclideanVariableLengthToyDataset(Dataset):
    def __init__(self, max_length=30):
        self.max_length = max_length
        self.vocab_size = 100
    def __getitem__(self, index):
        length = torch.randint(1, self.max_length + 1, (1,))
        mask = torch.ones(self.max_length, dtype=torch.bool)
        mask[length:] = False
        data = torch.arange(self.max_length, dtype=torch.float32) + torch.randn(self.max_length) * .01
        data = data * mask
        return {"data": data, "mask": mask}
    def __len__(self):
        return 10000

class MultimodalVariableLengthToyDataset(Dataset):
    def __init__(self, max_length=30, tokenizer: CharacterTokenizer = None):
        self.max_length = max_length
        self.vocab_size = 100
        self.labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        self.tokenizer = tokenizer
        self.text_vocab_size = len(self.labels)

    def __getitem__(self, index):
        length = torch.randint(1, self.max_length + 1, (1,))
        mask = torch.ones(self.max_length, dtype=torch.bool)
        mask[length:] = False
        data = torch.arange(self.max_length, dtype=torch.float32) + torch.randn(self.max_length) * .01
        data = data * mask
        
        # Get labels for the valid length
        current_len = length.item()
        labels_str = self.labels[:current_len]
        label_tokens = self.tokenizer.tokenize(labels_str)
        
        # Pad the rest
        if current_len < self.max_length:
            pad_id = self.tokenizer.char_to_idx['<pad>']
            padding = torch.full((self.max_length - current_len,), pad_id, dtype=torch.long)
            label = torch.cat((label_tokens, padding))
        else:
            label = label_tokens
            
        return {"data": data, "mask": mask, "label": label}
    def __len__(self):
        return 10000

def get_dataset(name, context_len=30, data_path='shakespeare.txt', max_length=30, tokenizer: CharacterTokenizer = None):
    if name == 'shakespeare':
        return ShakespeareDataset(context_len=context_len, data_path=data_path)
    elif name == 'euclidean_variable_length_toy':
        return EuclideanVariableLengthToyDataset(max_length=max_length)
    elif name == 'multimodal_variable_length_toy':
        return MultimodalVariableLengthToyDataset(max_length=max_length, tokenizer=tokenizer)
    else:
        print('Dataset is not implemented')
        return None