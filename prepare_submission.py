import argparse

import csv
import pandas


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a sentence-transformer model")

    # i/o arguments
    parser.add_argument(
        "--preds_filepath",
        type=str,
        required=True,
        help="path to .txt file containing predictions"
    )
    args = parser.parse_args()
    return args


def main():
    # parse arguments
    args = parse_args()

    # read dataset
    preds = pandas.read_csv(
        filepath_or_buffer=args.preds_filepath,
        sep="\t",
        header=None,
        names=["uid", "probs"]
    )
    preds["task_name"] = "ParaphraseIdentification"
    preds["probs"] = preds["probs"].apply(lambda x: 1 if x > 0.5 else 0).apply(lambda x: "P" if x == 1 else "NP")

    print(preds.columns)

    preds.rename(columns={"probs" : "label"}, inplace=True)

    print(preds.columns)

    preds["task_name"] = preds["task_name"].apply(lambda x: "\"" + str(x) + "\"")
    preds["uid"] = preds["uid"].apply(lambda x: "\"" + str(x) + "\"")
    preds["label"] = preds["label"].apply(lambda x: "\"" + str(x) + "\"")

    preds[["task_name", "uid", "label"]].to_csv("submission.txt", index=False, header=False, sep="\t", quoting=csv.QUOTE_NONE)

    print(preds.head())

if __name__ == "__main__":
    main()
