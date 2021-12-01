from enum import Enum
import transformers
import os
from pathlib import Path
import training

mod_path = Path(training.__file__).parent.parent
absolute_path = str(os.path.join(str(mod_path), "models"))

class TransformerType(Enum):
    BERT = 1
    ROBERTA = 2
    ELECTRA = 3

class Transformer:
    tokenizer: transformers.PreTrainedTokenizer = None
    config: transformers.PretrainedConfig = None
    model: transformers.PreTrainedModel = None

    def __init__(
            self,
            tokenizer: transformers.PreTrainedTokenizer,
            conf: transformers.PretrainedConfig,
            model: transformers.PreTrainedModel,
    ):
        self.tokenizer = tokenizer
        self.config = conf
        self.model = model

    @staticmethod
    def get_transformer(transformer_type: TransformerType):
        if transformer_type == TransformerType.BERT:
            conf = transformers.BertConfig(
                vocab_size=30_522,  # we align this to the tokenizer vocab_size
                max_position_embeddings=512,
                hidden_size=768,
                num_attention_heads=12,
                num_hidden_layers=6,
                type_vocab_size=1
            )
            return Transformer(
                tokenizer=transformers.BertTokenizer.from_pretrained(f"{absolute_path}/word_piece_tokenizer", max_len=512),
                conf=conf,
                model=transformers.BertForMaskedLM(conf)
            )
        elif transformer_type == TransformerType.ELECTRA:
            conf = transformers.ElectraConfig(
                vocab_size=30_522,  # we align this to the tokenizer vocab_size
                max_position_embeddings=512,  # 513?
                hidden_size=256,
                num_attention_heads=4,
                num_hidden_layers=6,  # this change is only because of computational reduction in prototype
                type_vocab_size=1  # this change makes sense (?)
            )
            return Transformer(
                tokenizer=transformers.ElectraTokenizer.from_pretrained(f"{absolute_path}/word_piece_tokenizer", max_len=512),
                conf=conf,
                model=transformers.ElectraForMaskedLM(conf)
            )
        elif transformer_type == TransformerType.ROBERTA:
            conf = transformers.RobertaConfig(
                vocab_size=30_522,  # we align this to the tokenizer vocab_size
                max_position_embeddings=512,
                hidden_size=768,
                num_attention_heads=12,
                num_hidden_layers=6,
                type_vocab_size=1
            )
            return Transformer(
                tokenizer=transformers.RobertaTokenizer.from_pretrained(f"{absolute_path}/byte_level_bpe_tokenizer", max_len=512),
                conf=conf,
                model=transformers.RobertaForMaskedLM(conf)
            )
        else:
            raise Exception(f"TransformerType {type} not implemented!")

#class BertTransformer(Transformer):






