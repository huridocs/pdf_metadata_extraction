"""Pre-download the large Flair NER model used by the text-to-text extractor methods.

The `ner-ontonotes-large` checkpoint is ~2.2 GB. When it is not cached, Flair downloads it
lazily during the first `create_model` run, which blocks that task for several minutes and
makes the end-to-end tests flaky.

Running this script during the image build (or before starting the services) keeps the
download out of the request critical path. Flair caches the model under `FLAIR_CACHE_ROOT`
(defaults to `~/.flair`).
"""

from flair.file_utils import hf_download

# Repository ids as resolved by Flair from the model key `ner-ontonotes-large`.
FLAIR_MODELS = [
    "flair/ner-english-ontonotes-large",
]


def main() -> None:
    for model in FLAIR_MODELS:
        print(f"Downloading Flair model: {model}")
        path = hf_download(model)
        print(f"Downloaded {model} to {path}")


if __name__ == "__main__":
    main()
