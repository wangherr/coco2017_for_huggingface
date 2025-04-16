import argparse

from datasets import load_dataset


def main(args):
    dataset = load_dataset("imagefolder", data_dir=args.image_dir, cache_dir=args.cache_dir)

    if args.caption_json.endswith('.json'):
        text_dataset = load_dataset('json', data_files=args.caption_json, field='caption', cache_dir=args.cache_dir)

        dataset['train'] = dataset['train'].add_column(name="text", column=text_dataset['train']['text'])

    if args.push:
        dataset.push_to_hub(args.dataset_name, private=True)

    return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image_dir",
        type=str,
        default="**/COCO/2017/train2017",
        help="",
    )
    parser.add_argument(
        "--caption_json",
        type=str,
        default="**/coco2017_train_caption.json",
        help="",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="results",
        help="",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="wangherr/coco2017_train_image_caption",
        help="",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="",
    )

    args = parser.parse_args()

    main(args)
