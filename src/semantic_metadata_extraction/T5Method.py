from os.path import join
from typing import List

from simpletransformers.config.model_args import T5Args
from simpletransformers.t5 import T5Model

from data.SemanticExtractionData import SemanticExtractionData
from semantic_metadata_extraction.Method import Method


class T5Method(Method):
    def performance(self, semantic_extraction_data: List[SemanticExtractionData]):
        if not semantic_extraction_data:
            return 0

        performance_train_set, performance_test_set = self.get_train_test(semantic_extraction_data)

        train_df = self.prepare_dataset()
        model_args = T5Args()
        model_args.max_seq_length = self.get_max_input_length(multilingual)
        model_args.max_length = self.get_max_output_length(multilingual)
        model_args.train_batch_size = 1
        model_args.eval_batch_size = 1
        model_args.num_train_epochs = 3
        model_args.evaluate_during_training = False
        model_args.evaluate_during_training_verbose = False
        model_args.evaluate_during_training_steps = 5000000
        model_args.use_multiprocessing = False
        model_args.use_multiprocessing_for_evaluation = False
        model_args.fp16 = False
        model_args.save_steps = -1
        model_args.use_cached_eval_features = False
        model_args.save_optimizer_and_scheduler = False
        model_args.save_eval_checkpoints = False
        model_args.save_model_every_epoch = False
        model_args.no_cache = True
        model_args.reprocess_input_data = False
        model_args.preprocess_inputs = False
        model_args.num_return_sequences = 1
        model_args.adafactor_eps = (1e-30, 5e-4)
        model_args.early_stopping_consider_epochs = False
        model_args.use_early_stopping = False
        model_args.manual_seed = 42
        model_args.overwrite_output_dir = True
        model_args.tensorboard_dir = f"{self.semantic_extraction_folder}/tensorboard_dir"
        model_args.output_dir = join(self.base_path, 't5')

        model = T5Model("t5", "t5-small", args=model_args, use_cuda=torch.cuda.is_available())

        model.train_model(train_df)

        correct = [index for index, test in enumerate(performance_test_set) if test.text == predictions[index]]
        return 100 * len(correct) / len(performance_test_set)

    def train(self, semantic_extraction_data: List[SemanticExtractionData]):
        pass

    def predict(self, texts: List[str]) -> List[str]:
        pass
