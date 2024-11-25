import datasets


def main():
    _dataset = datasets.load_dataset(
        "imagenet-1k",
        split="validation",
        trust_remote_code=True,
    )


if __name__ == "__main__":
    main()
