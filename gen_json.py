import argparse
import json

import tqdm
from torchvision.datasets import CocoDetection
from datasets import load_dataset


def main(args):
    dataset = CocoDetection(
        root=args.root,
        annFile=args.annFile)

    save_data = {
        "caption": []
    }
    for i in tqdm.tqdm(range(len(dataset))):
        _, labels = dataset[i]
        label_first = labels[0]
        caption = label_first['caption']
        line_dict = {'text': caption}

        save_data["caption"].append(line_dict)

    with open(args.out_path, 'w', encoding='utf-8') as json_file:
        json.dump(save_data, json_file, ensure_ascii=False, indent=4)

    dataset = load_dataset('json', data_files=args.out_path, field='caption', cache_dir=args.cache_dir)
    print(dataset['train'][0])

    if args.push:
        dataset.push_to_hub(args.dataset_name, private=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=str,
        default="**/COCO/2017/train2017",
        help="",
    )
    parser.add_argument(
        "--annFile",
        type=str,
        default="**/COCO/2017/annotations/captions_train2017.json",
        help="",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="results/cooc2017_train_caption.json",
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
        default="wangherr/coco2017_train_caption",
        help="",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="",
    )

    args = parser.parse_args()

    main(args)
