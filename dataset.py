##
import torch
import torchdata.datapipes as dp
import io
from PIL import Image
from torchvision import transforms
import os
from torchtext import vocab
from  torch.utils.data import DataLoader
import spacy
from torch.nn.functional import pad
from torch.utils.data.dataset import random_split
# Path to the images
FILE_PATH = "./sample.txt"

IMAGES_PATH = "Flicker8k_Dataset"

# Desired image dimensions
IMAGE_SIZE = (299, 299)

# Vocabulary size
VOCAB_SIZE = 10000

# Fixed length allowed for any sequence
SEQ_LENGTH = 25

# Dimension for the image embeddings and token embeddings
EMBED_DIM = 512

# Per-layer units in the feed-forward network
FF_DIM = 512

# Other training parameters
BATCH_SIZE = 64
VAL_BATCH_SIZE = 16
EPOCHS = 30
# AUTOTUNE = tf.data.AUTOTUNE

##
transform = transforms.Compose(
        [
            transforms.Resize((356, 356)),
            transforms.RandomCrop((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
##
def caption_read(tuple):
    line = tuple[1]
    line = line.strip()
    image_id, caption = line.split('\t')
    image_id = image_id.split("#")[0]
    print(caption)
    return image_id, caption

def open_image(x):
    return Image.open(os.path.join(IMAGES_PATH,x[0])), x[1]

def apply_transforms(x):
    return transform(x[0]), x[1]

def build_datapipes(path):
    new_dp = dp.iter.LineReader([('./Flickr8k.token.txt', io.open(path))])\
                     .map(caption_read) \
                     .enumerate()\
                     .to_map_datapipe()\
                     .map(open_image)\
                     .map(apply_transforms)
    return new_dp

def load_caption_data(filename):
    data_iter = build_datapipes(filename)
    for i, (image, caption) in enumerate(data_iter):
        print("==== coming here =====")
        print(image)
        print("======== done ========")
        break
##
def tokenize(text, tokenizer):
    test =  [tok.text for tok in tokenizer.tokenizer(text)]
    print("***** tokenizer", test)
    return test
##
def tokenize_en(text):
    return tokenize(text, spacy_eng)
##
def yield_tokens(data_iter, tokenizer, index):
    for tuple in data_iter:
        print("***** loader", tokenize(tuple[index], tokenizer))
        yield tokenize(tuple[index], tokenizer)
##
spacy_eng = spacy.load("en_core_web_lg")
def build_vocab(filepath):
    data_iter = dp.iter.LineReader([(filepath, io.open(filepath))]).map(caption_read)
    vocab_caption = vocab.build_vocab_from_iterator(yield_tokens(data_iter,spacy_eng, index=1),
                                                specials=["<s>", "</s>","<pad>","<unk>"])
    vocab_caption.set_default_index(vocab_caption["<unk>"])
    return vocab_caption
##
def collate_batch(batch, tokenizer, vocab, pad_idx):
    sos_idx =torch.tensor( vocab.get_stoi()["<s>"], dtype=torch.long)
    eos_idx = torch.tensor(vocab.get_stoi()["</s>"], dtype=torch.long)
    captions = []
    MAX_LEN = 0
    for idx,(image, caption) in enumerate(batch):
        tokens = [sos_idx] + vocab(tokenizer(caption)) + [eos_idx]
        MAX_LEN = max(MAX_LEN, len(tokens))
        captions.append(torch.tensor(tokens,dtype=torch.int64))
    print("***** MAX_LEN", MAX_LEN)
    for idx, tokens in enumerate(captions):
        captions[idx] = pad(tokens, (0, MAX_LEN - len(tokens)), value=pad_idx)
    print("***** captions", captions)
    return (torch.stack([image for image, caption in batch]), torch.stack(captions))

def create_dateloaders():
    vocab_caption = build_vocab(FILE_PATH)
    data_iter = build_datapipes(FILE_PATH)
    def collate_fn(batch):
       return collate_batch(batch, tokenize_en, vocab_caption, pad_idx=vocab_caption.get_stoi()["<pad>"])
    train_size = int(0.8 * len(data_iter))
    val_size = len(data_iter) - train_size
    train_dataset, val_dataset = random_split(data_iter, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    return train_loader, val_loader, vocab_caption

##
# print("==== building vocab ........")
# vocab_src = build_vocab('./sample.txt')
# print("******** len vocab_src ********", vocab_src.get_stoi())
# ##
# vocab_src(tokenize_en("hello left"))
##

##

