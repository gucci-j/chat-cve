from datasets import load_dataset
import re

def main(args):
    # Load the dataset
    print("Loading dataset...")
    dataset = load_dataset(
        "allenai/madlad-400",
        languages=[args.lang_code], 
        split="clean", 
        cache_dir=args.datasets_cache_dir,
        trust_remote_code=True
    )

    # Take 250K examples randomly
    if args.lang_code not in ("am", "my"):
        print("Taking 250K examples randomly...")
        dataset = dataset.shuffle(seed=42).select(range(250000))
    
    # Write the dataset to a .txt file
    print("Writing dataset to a .txt file...")
    if args.lang_code == "bn":
        split_ptn = r"[\n।\?!]"
    elif args.lang_code == "my":
        split_ptn = r"[\n။\?!]"
    else:
        split_ptn = r"[\n.\?!]"
    with open(args.output_file, "w") as f:
        lines = []
        for example in dataset:
            if lines == []:
                print([line.strip() for line in re.split(split_ptn, example["text"].replace("\\n", "\n")) if line.strip()])
            lines.extend([line.strip() for line in re.split(split_ptn, example["text"].replace("\\n", "\n")) if line.strip()])
        f.writelines([line + "\n" for line in lines])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lang_code",
        type=str,
        required=True,
        help="Language code of the dataset"
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        required=True,
        help="Output file path"
    )
    parser.add_argument(
        "--datasets_cache_dir",
        type=str,
        default="/home/username/datasets",
        help="Directory to cache the datasets"
    )
    args = parser.parse_args()
    main(args)
