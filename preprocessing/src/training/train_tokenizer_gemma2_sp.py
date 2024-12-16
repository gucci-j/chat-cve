import os

import sentencepiece as spm
from datasets import load_dataset
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
from transformers import GemmaTokenizer

def main(args):
    # train a sentencepiece model on it
    # the settings here are (best effort) those used for training Gemma 2
    options = dict(
        # input spec
        input=args.corpus_path, # path to the corpus
        input_format="text",
        # output spec
        model_prefix= args.output_dir + "target", # output filename prefix
        # algorithm spec
        # BPE alg
        model_type="bpe", # same as gemma-2
        vocab_size=args.vocab_size,
        # normalization
        normalization_rule_name="identity", # same as gemma-2
        remove_extra_whitespaces=False, # same as gemma-2
        input_sentence_size=0, # same as gemma-2
        max_sentence_length=4192,
        seed_sentencepiece_size=0, # same as gemma-2
        shuffle_input_sentence=True, # same as gemma-2
        # rare word treatment
        character_coverage=0.99995,
        byte_fallback=True, # same as gemma-2,
        # merge rules
        split_digits=True, # same as gemma-2
        split_by_unicode_script=True, # same as gemma-2
        split_by_whitespace=True, # same as gemma-2
        split_by_number=True, # same as gemma-2
        treat_whitespace_as_suffix=False, # same as gemma-2
        max_sentencepiece_length=16, # same as gemma-2
        add_dummy_prefix=False, # same as gemma-2
        allow_whitespace_only_pieces=True, 
        vocabulary_output_piece_score=True, # same as gemma-2
        hard_vocab_limit=True, # same as gemma-2
        use_all_vocab=False, # same as gemma-2
        user_defined_symbols=[ # same as gemma-2
            "<mask>",
            "<2mass>",
            "[@BOS@]",
            "<unused0>",
            "<unused1>",
            "<unused2>",
            "<unused3>",
            "<unused4>",
            "<unused5>",
            "<unused6>",
            "<unused7>",
            "<unused8>",
            "<unused9>",
            "<unused10>",
            "<unused11>",
            "<unused12>",
            "<unused13>",
            "<unused14>",
            "<unused15>",
            "<unused16>",
            "<unused17>",
            "<unused18>",
            "<unused19>",
            "<unused20>",
            "<unused21>",
            "<unused22>",
            "<unused23>",
            "<unused24>",
            "<unused25>",
            "<unused26>",
            "<unused27>",
            "<unused28>",
            "<unused29>",
            "<unused30>",
            "<unused31>",
            "<unused32>",
            "<unused33>",
            "<unused34>",
            "<unused35>",
            "<unused36>",
            "<unused37>",
            "<unused38>",
            "<unused39>",
            "<unused40>",
            "<unused41>",
            "<unused42>",
            "<unused43>",
            "<unused44>",
            "<unused45>",
            "<unused46>",
            "<unused47>",
            "<unused48>",
            "<unused49>",
            "<unused50>",
            "<unused51>",
            "<unused52>",
            "<unused53>",
            "<unused54>",
            "<unused55>",
            "<unused56>",
            "<unused57>",
            "<unused58>",
            "<unused59>",
            "<unused60>",
            "<unused61>",
            "<unused62>",
            "<unused63>",
            "<unused64>",
            "<unused65>",
            "<unused66>",
            "<unused67>",
            "<unused68>",
            "<unused69>",
            "<unused70>",
            "<unused71>",
            "<unused72>",
            "<unused73>",
            "<unused74>",
            "<unused75>",
            "<unused76>",
            "<unused77>",
            "<unused78>",
            "<unused79>",
            "<unused80>",
            "<unused81>",
            "<unused82>",
            "<unused83>",
            "<unused84>",
            "<unused85>",
            "<unused86>",
            "<unused87>",
            "<unused88>",
            "<unused89>",
            "<unused90>",
            "<unused91>",
            "<unused92>",
            "<unused93>",
            "<unused94>",
            "<unused95>",
            "<unused96>",
            "<unused97>",
            "<unused98>",
            "<start_of_turn>",
            "<end_of_turn>",
            "\n",
            "\n\n",
            "\n\n\n",
            "\n\n\n\n",
            "\n\n\n\n\n",
            "\n\n\n\n\n\n",
            "\n\n\n\n\n\n\n",
            "\n\n\n\n\n\n\n\n",
            "\n\n\n\n\n\n\n\n\n",
            "\n\n\n\n\n\n\n\n\n\n",
            "\n\n\n\n\n\n\n\n\n\n\n",
            "\n\n\n\n\n\n\n\n\n\n\n\n",
            "\n\n\n\n\n\n\n\n\n\n\n\n\n",
            "\n\n\n\n\n\n\n\n\n\n\n\n\n\n",
            "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n",
            "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n",
            "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n",
            "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n",
            "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n",
            "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n",
            "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n",
            "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n",
            "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n",
            "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n",
            "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n",
            "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n",
            "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n",
            "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n",
            "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n",
            "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n",
            "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n",
            "▁▁",
            "▁▁▁",
            "▁▁▁▁",
            "▁▁▁▁▁",
            "▁▁▁▁▁▁",
            "▁▁▁▁▁▁▁",
            "▁▁▁▁▁▁▁▁",
            "▁▁▁▁▁▁▁▁▁",
            "▁▁▁▁▁▁▁▁▁▁",
            "▁▁▁▁▁▁▁▁▁▁▁",
            "▁▁▁▁▁▁▁▁▁▁▁▁",
            "▁▁▁▁▁▁▁▁▁▁▁▁▁",
            "▁▁▁▁▁▁▁▁▁▁▁▁▁▁",
            "▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁",
            "▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁",
            "▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁",
            "▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁",
            "▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁",
            "▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁",
            "▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁",
            "▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁",
            "▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁",
            "▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁",
            "▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁",
            "▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁",
            "▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁",
            "▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁",
            "▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁",
            "▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁",
            "▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁",
            "<table>",
            "<caption>",
            "<thead>",
            "<tbody>",
            "<tfoot>",
            "<tr>",
            "<th>",
            "<td>",
            "</table>",
            "</caption>",
            "</thead>",
            "</tbody>",
            "</tfoot>",
            "</tr>",
            "</th>",
            "</td>",
            "<h1>",
            "<h2>",
            "<h3>",
            "<h4>",
            "<h5>",
            "<h6>",
            "<blockquote>",
            "</h1>",
            "</h2>",
            "</h3>",
            "</h4>",
            "</h5>",
            "</h6>",
            "</blockquote>",
            "<strong>",
            "<em>",
            "<b>",
            "<i>",
            "<u>",
            "<s>",
            "<sub>",
            "<sup>",
            "<code>",
            "</strong>",
            "</em>",
            "</b>",
            "</i>",
            "</u>",
            "</s>",
            "</sub>",
            "</sup>",
            "</code>",
        ], # same as gemma-2
        # special tokens (same as gemma-2)
        unk_id=3, # the UNK token MUST exist
        bos_id=2, # the others are optional, set to -1 to turn off
        eos_id=1,
        pad_id=0,
        # systems
        num_threads=os.cpu_count(), # use ~all system resources
    )
    spm.SentencePieceTrainer.train(**options)
    
    # load a model
    sp = spm.SentencePieceProcessor()
    sp.load(options["model_prefix"] + ".model")
    target_spm = sp_pb2_model.ModelProto()
    target_spm.ParseFromString(sp.serialized_model_proto())
    
    # load a original model
    base_sp_model = spm.SentencePieceProcessor()
    base_sp_model.Load(args.source_tokenizer_path)
    base_spm = sp_pb2_model.ModelProto()
    base_spm.ParseFromString(base_sp_model.serialized_model_proto())
    base_spm_tokens_set=set(p.piece for p in base_spm.pieces)
    
    # merge a tokenizer
    print(len(base_spm_tokens_set))
    print(f"Before:{len(base_spm_tokens_set)}")
    added_pieces = []
    new_count = 0
    for p in target_spm.pieces:
        piece = p.piece
        if piece not in base_spm_tokens_set and new_count < args.num_new_tokens:
            new_p = sp_pb2_model.ModelProto().SentencePiece()
            new_p.piece = piece
            new_p.score = 0
            base_spm.pieces.append(new_p)
            added_pieces.append(piece)
            new_count += 1
    print(f"New model pieces: {len(base_spm.pieces)}")
    
    # save
    os.makedirs(args.output_dir,exist_ok=True)
    with open(args.output_dir +'/merged.model', 'wb') as f:
        f.write(base_spm.SerializeToString())
    tokenizer = GemmaTokenizer(vocab_file=args.output_dir+'/merged.model')
    tokenizer.save_pretrained(args.output_dir)

    # iteratively remove newly added but unused tokens untill all new tokens are used
    num_iter = 0
    while True:
        ## load the dataset
        dataset = load_dataset(
            "text", 
            data_files={"train": [args.corpus_path]},
            split="train",
            cache_dir=args.cache_dir
        )

        ## tokenize the dataset
        dataset = dataset.map(
            lambda x: tokenizer(x["text"]),
            batched=True, remove_columns=dataset.column_names
        )

        ## get the token ids
        token_ids = set()
        for example in dataset:
            token_ids.update(example["input_ids"])

        ## remove the unused tokens
        num_removed = 0
        vocab = tokenizer.get_vocab()
        for p in base_spm.pieces:
            if vocab[p.piece] not in token_ids \
                and p.piece not in base_spm_tokens_set:
                base_spm.pieces.remove(p)
                num_removed += 1
        print(f"Removed {num_removed} unused tokens")
        if num_removed == 0 or num_iter > args.num_max_iter:
            ## save the updated model
            with open(args.output_dir +'/merged.model', 'wb') as f:
                f.write(base_spm.SerializeToString())
            tokenizer = GemmaTokenizer(vocab_file=args.output_dir+'/merged.model')
            tokenizer.save_pretrained(args.output_dir)
            break

        ## add the new tokens
        new_count = 0
        for p in target_spm.pieces:
            piece = p.piece
            if piece not in base_spm_tokens_set \
                and piece not in added_pieces \
                and new_count < num_removed:
                new_p = sp_pb2_model.ModelProto().SentencePiece()
                new_p.piece = piece
                new_p.score = 0
                base_spm.pieces.append(new_p)
                added_pieces.append(piece)
                new_count += 1
        print(f"New model pieces: {len(base_spm.pieces)}")

        ## save the updated model
        with open(args.output_dir +'/merged.model', 'wb') as f:
            f.write(base_spm.SerializeToString())
        tokenizer = GemmaTokenizer(vocab_file=args.output_dir+'/merged.model')
        tokenizer.save_pretrained(args.output_dir)

        num_iter += 1


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus_path", 
        type=str,
        help="Path to the corpus to train the tokenizer on",
        required=True
    )
    parser.add_argument(
        "--vocab_size", 
        type=int,
        help="Vocabulary size of the tokenizer",
        required=True
    )
    parser.add_argument(
        "--source_tokenizer_path", 
        type=str,
        help="Path to the source tokenizer",
        required=True
    )
    parser.add_argument(
        "--output_dir", 
        type=str,
        help="Path to the output directory",
        required=True
    )
    parser.add_argument(
        "--lang_code", 
        type=str,
        help="Language code",
        required=True,
        choices=["am", "bn", "gu", "my", "si", "ta", "te"]
    )
    parser.add_argument(
        "--num_new_tokens", 
        type=int,
        help="Number of new tokens to add to the tokenizer",
        default=100
    )
    parser.add_argument(
        "--num_max_iter", 
        type=int,
        help="Maximum number of iterations to remove unused tokens",
        default=10
    )
    parser.add_argument(
        "--cache_dir", 
        type=str,
        help="Path to the cache directory",
    )
    args = parser.parse_args()
    main(args)
    