from tokenizers import Tokenizer
from tokenizers.models import WordPiece

from tokenizers import normalizers
from tokenizers.normalizers import NFD
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

from tokenizers.trainers import WordPieceTrainer

# Training a WordPiece tokenizer because MBERT is also trained on WordPiece
bert_tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))

# Normalizing the text by unicode characters, no capital letters in Kannada, so no need for making lowercase, and no accent stripping
bert_tokenizer.normalizer = normalizers.Sequence([NFD()])

# Splitting text on spaces (isolating words)
bert_tokenizer.pre_tokenizer = Whitespace()

# Adding special characters for training BERT
bert_tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[
        ("[CLS]", 1),
        ("[SEP]", 2),
    ],
)

trainer = WordPieceTrainer(vocab_size=30522, min_frequency=2, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
files = ["data/kn.txt"]
bert_tokenizer.train(files, trainer)
bert_tokenizer.save("tokenizer/out/tokenizer.json")

