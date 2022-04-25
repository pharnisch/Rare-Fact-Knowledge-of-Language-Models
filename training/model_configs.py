from enum import Enum
import transformers
import os
from pathlib import Path
#import training

mod_path = Path(__file__).parent.parent
absolute_path = str(os.path.join(str(mod_path), "models"))

class TransformerType(Enum):
    BERT = 1
    ROBERTA = 2
    ELECTRA = 3
    BERT_PRETRAIN = 4

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
    def get_transformer(transformer_type: TransformerType, num_hidden_layers=12):
        if transformer_type == TransformerType.BERT:
            conf = transformers.BertConfig(
                vocab_size=30_522,  # we align this to the tokenizer vocab_size
                max_position_embeddings=512,
                hidden_size=768,
                num_attention_heads=12,
                num_hidden_layers=num_hidden_layers,
                type_vocab_size=1
            )
            return Transformer(
                tokenizer=transformers.BertTokenizerFast.from_pretrained(f"{absolute_path}/word_piece_tokenizer", model_max_length=512),
                conf=conf,
                model=transformers.BertForMaskedLM(conf)
            )
        elif transformer_type == TransformerType.ELECTRA:
            conf = transformers.ElectraConfig(
                vocab_size=30_522,  # we align this to the tokenizer vocab_size
                max_position_embeddings=512,
                hidden_size=256,
                num_attention_heads=4,
                num_hidden_layers=num_hidden_layers,
                type_vocab_size=1  # this change makes sense (?) because we only insert one input (no SEP and more content)
            )
            return Transformer(
                tokenizer=transformers.ElectraTokenizerFast.from_pretrained(f"{absolute_path}/word_piece_tokenizer", model_max_length=512),
                conf=conf,
                model=transformers.ElectraForMaskedLM(conf)
            )
        elif transformer_type == TransformerType.ROBERTA:
            conf = transformers.RobertaConfig(
                vocab_size=30_522,  # we align this to the tokenizer vocab_size
                max_position_embeddings=514,   # braucht scheinbar 2 mehr!!! sonst error
                hidden_size=768,
                num_attention_heads=12,
                num_hidden_layers=num_hidden_layers,
                type_vocab_size=1
            )
            return Transformer(
                tokenizer=transformers.RobertaTokenizerFast.from_pretrained(f"{absolute_path}/byte_level_bpe_tokenizer", model_max_length=512),
                conf=conf,
                model=transformers.RobertaForMaskedLM(conf)
            )
        elif transformer_type == TransformerType.BERT_PRETRAIN:
            conf = transformers.BertConfig(
                vocab_size=30_522,  # we align this to the tokenizer vocab_size
                max_position_embeddings=512,
                hidden_size=768,
                num_attention_heads=12,
                num_hidden_layers=num_hidden_layers,
                type_vocab_size=1
            )
            return Transformer(
                tokenizer=transformers.BertTokenizerFast.from_pretrained(f"{absolute_path}/word_piece_tokenizer",
                                                                     model_max_length=512),
                conf=conf,
                model=transformers.BertForPreTraining(conf)
            )
        else:
            raise Exception(f"TransformerType {type} not implemented!")


#class BertTransformer(Transformer):






