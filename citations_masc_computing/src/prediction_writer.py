from pytorch_lightning.callbacks import BasePredictionWriter
from utilities import extract_triplets_rcv2
import json

class PredictionWriter(BasePredictionWriter):

    def __init__(self, output_file, write_interval):
        super().__init__(write_interval)
        self.output_file = output_file

    def write_on_batch_end(
        self, trainer,
            pl_module,
            prediction,
            batch_indices,
            batch,
            batch_idx,
            dataloader_idx
    ):
        # TODO: TEST THIS
        # decoded inputs
        generated_tokens = self.model.generate(
            batch["input_ids"].to(self.model.device),
            attention_mask=batch["attention_mask"].to(self.model.device),
            # use_cache=True,
            # **gen_kwargs,
        )
        decoded_inputs = pl_module.tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)

        # outputs (predictions)
        preds_per_batch = [batch for something in prediction for batch in something]
        parsed_preds = [extract_triplets_rcv2(rel) for batch in preds_per_batch for rel in batch]

        local_data = []
        for input_text, prediction in zip(decoded_inputs, parsed_preds):
            local_data.append({"input": input_text, "prediction": prediction})
        # Save to local file
        i = 0
        with open(self.output_file, 'a') as f:
            print("saving to json (batch end)..........................\n")
            for item in local_data:
                cleaned_item = {}
                cleaned_item["input"] = clean_text(item["input"])  # this is a single sequence
                cleaned_item["prediction"] = item["prediction"]
                i += 1
                # we have to do the coupling by ourselves then ?
                f.write(json.dumps(cleaned_item) + '\n')

    def write_on_epoch_end(self,
                           trainer,
                           pl_module,
                           predictions,
                           batch_indices):
        # Writes predictions sequentially, one line per input (multiple relations per input)
        preds_per_batch = [batch for something in predictions for batch in something]
        parsed_preds = [extract_triplets_rcv2(rel) for batch in preds_per_batch for rel in batch]
        with open(self.output_file, 'a', encoding="utf8") as f:
            print("saving to json (epoch end)..........................\n")
            for item in parsed_preds:
                cleaned_item = {}
                cleaned_item["predictions"] = item
                f.write(json.dumps(cleaned_item) + '\n')


def clean_text(text):
    # Decode Unicode escape sequences

    # Remove padding and repetitive sequences
    text = text.replace('<pad>', '')
    text = text.replace('tp_XX', '')  # Remove <pad> tokens
    text = ' '.join(text.split())  # Remove extra spaces and newlines

    return text