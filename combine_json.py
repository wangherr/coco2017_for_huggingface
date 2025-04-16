import argparse

from datasets import load_dataset, DatasetDict


def main(args):
    train_dataset = load_dataset('json', data_files=args.train_json, field='caption')['train']
    val_dataset = load_dataset('json', data_files=args.val_json, field='caption')['train']

    dataset = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset
    })

    if args.push:
        dataset.push_to_hub(args.dataset_name, private=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_json",
        type=str,
        default="results/cooc2017_train_caption.json",
        help="",
    )
    parser.add_argument(
        "--val_json",
        type=str,
        default="results/cooc2017_validation_caption.json",
        help="",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="wangherr/coco2017_caption",
        help="",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="",
    )

    args = parser.parse_args()

    main(args)
