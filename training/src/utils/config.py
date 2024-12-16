import argparse
from transformers import HfArgumentParser, TrainingArguments

class CustomArgumentParser(argparse.ArgumentParser):
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="Tune a language model."
        )
        self.hf_parser = HfArgumentParser(TrainingArguments)

        # Define any custom arguments using argparse
        self.parser.add_argument(
            "--dataset_path",
            type=str,
            required=True,
            help="Path to the tokenized dataset."
        )
        self.parser.add_argument(
            "--val_dataset_path",
            type=str,
            default=None,
            help="Path to the tokenized validation dataset."
        )
        self.parser.add_argument(
            "--mix_english_data",
            action="store_true",
            help="Whether to mix English data with the target language data."
        )
        self.parser.add_argument(
            "--english_dataset_path",
            type=str,
            default=None,
            help="Path to the English dataset."
        )
        self.parser.add_argument(
            "--tokenizer_name_or_path", 
            type=str, 
            required=True,
            help="Path to the tokenizer."
        )
        self.parser.add_argument(
            "--model_name_or_path", 
            type=str, 
            required=True,
            help="Path to the model."
        )
        self.parser.add_argument(
            "--cache_dir", 
            type=str, 
            default=None,
            help="Path to the cache directory."
        )
        self.parser.add_argument(
            "--model_type", 
            type=str, 
            required=True,
            choices=["llama3", "gemma2", "qwen2"],
            help="Model type."
        )
        self.parser.add_argument(
            "--copy_lm_head",
            action="store_true",
            help="Whether to copy LM head."
        )
        self.parser.add_argument(
            "--is_baseline",
            action="store_true",
            help="Set this to train a model with the baseline settings."
        )
        self.parser.add_argument(
            "--num_lm_heads",
            type=int,
            default=1,
            help="The number of LM heads."
        )

    def parse_args(self):
        args, extras = self.parser.parse_known_args()
        training_args = self.hf_parser.parse_args_into_dataclasses(extras)[0]
        return args, training_args
    