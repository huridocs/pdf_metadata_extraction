import tensorflow_hub

# Do not remove the next import as it is needed for the models from tensorflow-hub
import tensorflow_text as text

module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
sentence_embeddings_model = tensorflow_hub.load(module_url)

multilingual_module_url = "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"
multilingual_sentence_embeddings_model = tensorflow_hub.load(multilingual_module_url)
