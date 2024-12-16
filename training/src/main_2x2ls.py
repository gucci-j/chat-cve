import logging
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import datasets
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForLanguageModeling, Trainer,
                          AutoConfig)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from utils import CustomArgumentParser
from models import LlamaForMultiCausalLM, Gemma2ForMultiCausalLM


def main(args, training_args):
    #####
    # Load the dataset
    #####
    train_dataset = datasets.load_from_disk(args.dataset_path)
    if args.mix_english_data:
        print("Mixing English data...")
        # Mix the English data with the target language data
        english_dataset = datasets.load_from_disk(args.english_dataset_path)
        # Sample 10% of train_dataset from english_dataset
        english_dataset = english_dataset.shuffle(seed=training_args.seed).select(
            range(math.ceil(0.1 * len(train_dataset)))
        )
        # Concatenate the two datasets
        train_dataset = datasets.concatenate_datasets([train_dataset, english_dataset])
        print("Done!")
    train_dataset = train_dataset.shuffle(seed=training_args.seed)
    if args.val_dataset_path is not None:
        val_dataset = datasets.load_from_disk(args.val_dataset_path)
    else:
        val_dataset = None

    #####
    # Load the tokenizer
    #####
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name_or_path,
        cache_dir=args.cache_dir
    )
    tokenizer.pad_token = tokenizer.eos_token

    #####
    # Set up the data collator
    #####
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    #####
    # Load the model
    #####
    if args.is_baseline:
        # - No MTP
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            device_map="cuda", 
            cache_dir=args.cache_dir
        )
    else:
        # - With MTP
        config = AutoConfig.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir
        )
        config.num_lm_heads = args.num_lm_heads
        if args.copy_lm_head:
            config.copy_lm_head = True
        else:
            config.copy_lm_head = False
        if args.model_type == "gemma2":
            model = Gemma2ForMultiCausalLM.from_pretrained(
                args.model_name_or_path,
                device_map="cuda",
                cache_dir=args.cache_dir,
                config=config
            )
        elif args.model_type == "llama3":
            model = LlamaForMultiCausalLM.from_pretrained(
                args.model_name_or_path,
                device_map="cuda",
                cache_dir=args.cache_dir,
                config=config
            )
        if config.copy_lm_head:
            for i in range(config.num_lm_heads):
                with torch.no_grad():
                    model.lm_heads[i].weight.copy_(model.lm_head.weight)
    # Freeze all layers except 2x2 LS
    for param in model.model.layers.parameters():
        param.requires_grad = False
    for index in [0, 1, -2, -1]:
        for param in model.model.layers[index].parameters():
            param.requires_grad = True
    logger.info(model)

    #####
    # Set up the trainer
    #####
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        eval_dataset=val_dataset,
    )
    
    #####
    # Train the model
    #####
    trainer.train()

    #####
    # Save the model
    #####
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    parser = CustomArgumentParser()
    args, training_args = parser.parse_args()
    logger.info(args)
    logger.info(training_args)

    main(args, training_args)
