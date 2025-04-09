from typing import Optional
import json
from omegaconf import DictConfig

from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    default_data_collator,
    set_seed,
    DataCollatorWithPadding
)


class InferencePLDataModule(pl.LightningDataModule):
    def __init__(self, conf: DictConfig, tokenizer: AutoTokenizer):
        super().__init__()
        self.conf = conf
        self.tokenizer = tokenizer
        # self.id_column = conf.id_column
        # self.text_column = conf.text_column
        self.prefix = conf.source_prefix if conf.source_prefix is not None else ""
        # this function is heavily reduced from the original (it might cause some isssue)

    def prepare_data(self):
        pass


    def inference_dataloader(self, inputs: list) -> DataLoader:
        # New method to create a DataLoader for inference
        # Converts a list of inputs into a dataset
        inference_dataset = self._create_inference_dataset(inputs)
        return DataLoader(
            inference_dataset,
            batch_size=self.conf.eval_batch_size,  # Using eval batch size or define a new one for inference
            # collate_fn=default_data_collator,
            collate_fn=DataCollatorWithPadding(tokenizer=self.tokenizer, return_tensors="pt"),
            num_workers=self.conf.dataloader_num_workers,
            pin_memory=self.conf.dataloader_pin_memory,
        )

    def _create_inference_dataset(self, inputs: list) -> Dataset:
        # Convert inputs to a PyTorch Dataset
        class InferenceDataset(Dataset):
            def __init__(self, tokenizer, inputs, prefix, max_source_length):
                self.tokenizer = tokenizer
                self.inputs = inputs
                self.prefix = prefix
                self.max_source_length = max_source_length
                # TODO: how to keep idx in Dataloader ?

            def __len__(self):
                return len(self.inputs)

            def __getitem__(self, idx):
                input_text = self.prefix + self.inputs[idx]['text']
                return self.tokenizer(input_text, truncation=True,
                                      padding=True,
                                      max_length=self.max_source_length)

        return InferenceDataset(self.tokenizer, inputs, self.prefix, self.conf.max_source_length)


    @staticmethod
    def load_texts_from_jsonl(file_path):
        inputs = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                # Parse the JSON object from each line
                json_obj = json.loads(line)
                # Extract the text column and add it to the inputs list
                if "text" in json_obj:
                    inputs.append({"id":json_obj["id"],"text": json_obj["text"]})
                else:
                    raise KeyError("There is no 'text' field in json file. Please change field name.")
        return inputs
