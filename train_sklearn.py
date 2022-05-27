import argparse

import pandas
import sklearn.feature_extraction
import sklearn.linear_model
import sklearn.model_selection
import sklearn.svm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a sentence-transformer model")

    # i/o arguments
    parser.add_argument(
        "--train_filepath",
        type=str,
        required=True,
        help="path to .csv file containing training data"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="directory in which to cache fetched pretrained models"
    )

    # reproducibility arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=2022,
        help="random seed"
    )
    args = parser.parse_args()
    return args


def main():
    # parse arguments
    args = parse_args()

    # read data
    print("Loading dataset...")
    data = pandas.read_csv(filepath_or_buffer=args.train_filepath, sep="\t")

    # process data
    # data["text"] = data["text"].apply(preprocessing.remove_final_period)
    # data["text"] = data["text"].apply(preprocessing.lowercase_text)

    # convert text to features
    print("Preparing training features...")
    featurizer = sklearn.feature_extraction.text.TfidfVectorizer(lowercase=False)
    X_train = featurizer.fit_transform(data["text"])

    print(f"Features: {len(featurizer.vocabulary_)}")

    # run training
    # model = sklearn.linear_model.LogisticRegression()
    model = sklearn.svm.SVC()

    search_space = {"C": [0.1, 1]}

    grid_model = sklearn.model_selection.GridSearchCV(
        estimator=model,
        param_grid=search_space,
        scoring="accuracy",
        n_jobs=None,
        refit=True,
        cv=5,
        verbose=0
    )

    print("Training...")
    grid_model.fit(X_train, data["target"])

    idx = grid_model.best_index_
    avg_score = round(grid_model.best_score_, 3)
    std_score = round(grid_model.cv_results_["std_test_score"][idx], 3)
    worst_score = round(min(grid_model.cv_results_[f"split{i}_test_score"][idx] for i in range(grid_model.cv)), 3)
    best_score = round(max(grid_model.cv_results_[f"split{i}_test_score"][idx] for i in range(grid_model.cv)), 3)
    score_range = round(best_score - worst_score, 3)

    print(
        f"Average best score is {avg_score} with std {std_score}. \n"
        f"Worst fold has a score of {worst_score}, \n"
        f"best fold has a score of {best_score}. \n"
        f"Range: {score_range}")


if __name__ == "__main__":
    main()
