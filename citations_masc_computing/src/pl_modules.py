from typing import Any
import nltk
import json
import pytorch_lightning as pl
import torch
import numpy as np
import pandas as pd
from score import score, re_score
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.optimization import (
    Adafactor,
    AdamW,
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup
)
from scheduler import get_inverse_square_root_schedule_with_warmup
from datasets import load_metric
from utilities import shift_tokens_left, extract_triplets, extract_triplets_rcv2
from utils.dataset_utils import REL_LABEL2ID as relations_rcv2
from dataset_builders import rcv2 as rcv2_dataset

arg_to_scheduler = {
    "linear": get_linear_schedule_with_warmup,
    "cosine": get_cosine_schedule_with_warmup,
    "cosine_w_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
    "polynomial": get_polynomial_decay_schedule_with_warmup,
    "constant": get_constant_schedule,
    "constant_w_warmup": get_constant_schedule_with_warmup,
    "inverse_square_root": get_inverse_square_root_schedule_with_warmup
}

class BasePLModule(pl.LightningModule):

    def __init__(self, conf, config: AutoConfig, tokenizer: AutoTokenizer, model: AutoModelForSeq2SeqLM, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(conf)
        self.tokenizer = tokenizer
        self.model = model
        self.config = config
        self.conf=conf
        if self.model.config.decoder_start_token_id is None:
            raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

        if self.hparams.label_smoothing == 0:
            self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
        else:
            # dynamically import label_smoothed_nll_loss
            from utilities import label_smoothed_nll_loss

            self.loss_fn = label_smoothed_nll_loss

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, conf, config, tokenizer, model, **kwargs):
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)

        # Instantiate the class
        instance = cls(conf=conf,config=config, tokenizer=tokenizer, model=model, **kwargs)

        # Load model state dict
        instance.load_state_dict(checkpoint["state_dict"])

        return instance

    def forward(self, inputs, labels, **kwargs) -> dict:
        """
        Method for the forward pass.
        'training_step', 'validation_step' and 'test_step' should call
        this method in order to compute the output predictions and the loss.

        Returns:
            output_dict: forward output containing the predictions (output logits ecc...) and the loss if any.

        """
        if self.hparams.label_smoothing == 0:
            if self.hparams is not None and self.hparams.ignore_pad_token_for_loss:
                # force training to ignore pad token
                outputs = self.model(**inputs, use_cache=False, return_dict = True, output_hidden_states=True)
                logits = outputs['logits']
                loss = self.loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))#, ignore_index=self.config.pad_token_id)
            else:
                # compute usual loss via models
                outputs = self.model(**inputs, labels=labels, use_cache=False, return_dict = True, output_hidden_states=True)
                loss = outputs['loss']
                logits = outputs['logits']
        else:
            # compute label smoothed loss
            outputs = self.model(**inputs, use_cache=False, return_dict = True, output_hidden_states=True)
            logits = outputs['logits']
            lprobs = torch.nn.functional.log_softmax(logits, dim=-1)
            # labels = torch.where(labels != -100, labels, self.config.pad_token_id)
            labels.masked_fill_(labels == -100, self.config.pad_token_id)
            loss, _ = self.loss_fn(lprobs, labels, self.hparams.label_smoothing, ignore_index=self.config.pad_token_id)
        output_dict = {'loss': loss, 'logits': logits}
        # return loss, logits
        return output_dict

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        labels = batch.pop("labels")
        labels_original = labels.clone()
        batch["decoder_input_ids"] = torch.where(labels != -100, labels, self.config.pad_token_id)
        labels = shift_tokens_left(labels, -100)
        forward_output = self.forward(batch, labels)
        self.log('loss', forward_output['loss'])
        batch["labels"] = labels_original
        if 'loss_aux' in forward_output:
            self.log('loss_classifier', forward_output['loss_aux'])
            return forward_output['loss'] + forward_output['loss_aux']
        return forward_output['loss']# + forward_output['loss_aux']

    def _pad_tensors_to_max_len(self, tensor, max_length):
        # If PAD token is not defined at least EOS token has to be defined
        pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else self.config.eos_token_id

        if pad_token_id is None:
            raise ValueError(
                f"Make sure that either `config.pad_token_id` or `config.eos_token_id` is defined if tensor has to be padded to `max_length`={max_length}"
            )

        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor

    def generate_triples(self,
        batch,
        labels,
    ) -> None:
        gen_kwargs = {
            "max_length": self.hparams.val_max_target_length
            if self.hparams.val_max_target_length is not None
            else self.config.max_length,
            "early_stopping": False,
            "length_penalty": 0,
            "no_repeat_ngram_size": 0,
            "num_beams": self.hparams.eval_beams if self.hparams.eval_beams is not None else self.config.num_beams,
        }

        generated_tokens = self.model.generate(
            batch["input_ids"].to(self.model.device),
            attention_mask=batch["attention_mask"].to(self.model.device),
            use_cache = True,
            **gen_kwargs,
        )

        decoded_preds = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)
        decoded_labels = self.tokenizer.batch_decode(torch.where(labels != -100, labels, self.config.pad_token_id), skip_special_tokens=False)

        if self.hparams.dataset_name.split('/')[-1] == 'rcv2.py':
            return [extract_triplets_rcv2(rel) for rel in decoded_preds], [extract_triplets_rcv2(rel) for rel in decoded_labels]

        return [extract_triplets(rel) for rel in decoded_preds], [extract_triplets(rel) for rel in decoded_labels]

    def generate_samples(self,
        # model,
        # tokenizer,
        batch,
        labels,
    ) -> None:
        # labels = batch.pop("labels")
        # pick the last batch and logits
        # x, y = batch
        gen_kwargs = {
            "max_length": self.hparams.val_max_target_length
            if self.hparams.val_max_target_length is not None
            else self.config.max_length,
            "early_stopping": False,
            "length_penalty": 0,
            "no_repeat_ngram_size": 0,
            "num_beams": self.hparams.eval_beams if self.hparams.eval_beams is not None else self.config.num_beams,
        }
        relation_start = labels == 50265
        relation_start = torch.roll(relation_start, 1, 1)
        relation_start = torch.cumsum(relation_start, dim=1)
        labels_decoder = torch.where(relation_start == 1, self.tokenizer.pad_token_id, labels)
        labels_decoder[:,-1] = 2
        labels_decoder = torch.roll(labels_decoder, 1, 1)

        generated_tokens = self.model.generate(
            batch["input_ids"].to(self.model.device),
            attention_mask=batch["attention_mask"].to(self.model.device),
            decoder_input_ids=labels_decoder.to(self.model.device),
            use_cache = False,
            **gen_kwargs,
        )
        relation_start = generated_tokens == 50265
        relation_start = torch.roll(relation_start, 2, 1)

        decoded_preds = self.tokenizer.batch_decode(generated_tokens[relation_start==1], skip_special_tokens=False)

        return [rel.strip() for rel in decoded_preds]

    def forward_samples(self,
        # model,
        # tokenizer,
        batch,
        labels,
    ) -> None:
        relation_start = labels == 50265
        relation_start = torch.roll(relation_start, 2, 1)
        labels = torch.where(torch.cumsum(relation_start, dim=1) == 1, self.tokenizer.pad_token_id, labels)
        labels[:,-1] = 0
        labels = torch.roll(labels, 1, 1)
        min_padding = min(torch.sum((labels == 1).int(), 1))
        labels_decoder = torch.randint(60000,(labels.shape[0], labels.shape[1] - min_padding))
        labels_decoder = labels[:, :-min_padding]
        outputs = self.model(
            batch["input_ids"].to(self.model.device),
            attention_mask=batch["attention_mask"].to(self.model.device),
            decoder_input_ids=labels_decoder.to(self.model.device),
            return_dict=True,
        )
        next_token_logits = outputs.logits[relation_start[:,: -min_padding]==1]
        next_tokens = torch.argmax(next_token_logits, dim=-1)

        decoded_preds = self.tokenizer.batch_decode(next_tokens, skip_special_tokens=False)

        return [rel.strip() for rel in decoded_preds]

    def validation_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0) -> None:
        gen_kwargs = {
            "max_length": self.hparams.val_max_target_length
            if self.hparams.val_max_target_length is not None
            else self.config.max_length,
            "early_stopping": False,
            "no_repeat_ngram_size": 0,
            "length_penalty": 0,
            "num_beams": self.hparams.eval_beams if self.hparams.eval_beams is not None else self.config.num_beams,
        }

        if self.hparams.predict_with_generate and not self.hparams.prediction_loss_only:
            generated_tokens = self.model.generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                **gen_kwargs,
            )
            # in case the batch is shorter than max length, the output should be padded
            if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
                generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

        labels = batch.pop("labels")
        batch["decoder_input_ids"] = torch.where(labels != -100, labels, self.config.pad_token_id)
        labels = shift_tokens_left(labels, -100)
        with torch.no_grad():
            # compute loss on predict data
            forward_output = self.forward(batch, labels)

        forward_output['loss'] = forward_output['loss'].mean().detach()

        if self.hparams.prediction_loss_only:
            self.log('val_loss', forward_output['loss'])
            return

        forward_output['logits'] = generated_tokens.detach() if self.hparams.predict_with_generate else forward_output['logits'].detach()

        if labels.shape[-1] < gen_kwargs["max_length"]:
            forward_output['labels'] = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])
        else:
            forward_output['labels'] = labels

        if self.hparams.predict_with_generate:
            metrics = self.compute_metrics(forward_output['logits'].detach().cpu(), forward_output['labels'].detach().cpu())
        else:
            metrics = {}
        metrics['val_loss'] = forward_output['loss']
        for key in sorted(metrics.keys()):
            self.log(key, metrics[key])

        outputs = {}
        outputs['predictions'], outputs['labels'] = self.generate_triples(batch, labels)
        return outputs

    def test_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0) -> None:
        gen_kwargs = {
            "max_length": self.hparams.val_max_target_length
            if self.hparams.val_max_target_length is not None
            else self.config.max_length,
            "early_stopping": self.hparams.apply_early_stopping,
            "no_repeat_ngram_size": 0,
            "length_penalty": 0,
            "num_beams": self.hparams.eval_beams if self.hparams.eval_beams is not None else self.config.num_beams,
        }

        if self.hparams.predict_with_generate and not self.hparams.prediction_loss_only:
            generated_tokens = self.model.generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                **gen_kwargs,
            )
            # in case the batch is shorter than max length, the output should be padded
            if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
                generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])
        labels = batch.pop("labels")
        batch["decoder_input_ids"] = torch.where(labels != -100, labels, self.config.pad_token_id)
        labels = shift_tokens_left(labels, -100)
        with torch.no_grad():
            # compute loss on predict data
            forward_output = self.forward(batch, labels)

        forward_output['loss'] = forward_output['loss'].mean().detach()
        if self.hparams.prediction_loss_only:
            self.log('test_loss', forward_output['loss'])
            return

        forward_output['logits'] = generated_tokens.detach() if self.hparams.predict_with_generate else forward_output['logits'].detach()

        if labels.shape[-1] < gen_kwargs["max_length"]:
            forward_output['labels'] = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])
        else:
            forward_output['labels'] = labels

        if self.hparams.predict_with_generate:
            metrics = self.compute_metrics(forward_output['logits'].detach().cpu(), forward_output['labels'].detach().cpu())
        else:
            metrics = {}
        metrics['test_loss'] = forward_output['loss']
        for key in sorted(metrics.keys()):
            self.log(key, metrics[key], prog_bar=True)

        if self.hparams.finetune:
            return {'predictions': self.forward_samples(batch, labels)}
        else:
            outputs = {}
            outputs['predictions'], outputs['labels'] = self.generate_triples(batch, labels)
            return outputs

    def validation_epoch_end(self, output: dict) -> Any:
        if self.hparams.relations_file:
            relations_df = pd.read_csv(self.hparams.relations_file, header = None, sep='\t')
            relations = list(relations_df[0])
            scores, precision, recall, f1 = re_score([item for pred in output for item in pred['predictions']], [item for pred in output for item in pred['labels']], relations)
            self.log('val_prec_micro', precision)
            self.log('val_recall_micro', recall)
            self.log('val_F1_micro', f1)
        elif self.hparams.dataset_name.split('/')[-1] == 'rcv2.py':
                # TODO: Change if changed extract_triplets_rcv2
                scores, precision, recall, f1 = re_score(
                    [item for pred in output for item in pred['predictions']],
                    [item for pred in output for item in pred['labels']],
                    entity_types=list(rcv2_dataset.TAG2ENTITY.keys()),
                    relation_types=list(rcv2_dataset.REL_LABEL2TEXT.values()),
                    mode=self.hparams.scoring_mode)
        else:
            key = []
            preds_list = []
            labels_list = []
            for ele in output:
                for pred, lab in zip(ele['predictions'], ele['labels']):
                    if len(pred) == 0 or len(lab) == 0:
                        continue
                    preds_list.append(pred[0]["type"])
                    labels_list.append(lab[0]["type"])
            prec_micro, recall_micro, f1_micro = score(labels_list, preds_list, verbose=True)
            self.log('val_prec_micro', prec_micro)
            self.log('val_recall_micro', recall_micro)
            self.log('val_F1_micro', f1_micro)

    def test_epoch_end(self, output: dict) -> Any:
        if not self.hparams.finetune and self.hparams.relations_file:
            relations_df = pd.read_csv(self.hparams.relations_file, header = None, sep='\t')
            relations = list(relations_df[0])
            scores, precision, recall, f1 = re_score([item for pred in output for item in pred['predictions']], [item for pred in output for item in pred['labels']], relations)
            self.log('test_prec_micro', precision)
            self.log('test_recall_micro', recall)
            self.log('test_F1_micro', f1)
        elif self.hparams.dataset_name.split('/')[-1] == 'rcv2.py':
                print("\nScoring on test ..........")
                scores, precision, recall, f1 = re_score(
                    [item for pred in output for item in pred['predictions']],
                    [item for pred in output for item in pred['labels']],
                    entity_types=list(rcv2_dataset.TAG2ENTITY.keys()),
                    relation_types=list(rcv2_dataset.REL_LABEL2TEXT.values()),
                    mode=self.hparams.scoring_mode)
        else:
            preds_list = []
            labels_list = []
            i = 0
            with open("preds.jsonl", "w") as f:
                for ele in output:
                    for pred, lab in zip(ele['predictions'], ele['labels']):
                        if len(pred) == 0 or len(lab) == 0:
                            continue
                        f.write(f'{pred[0]} \t {lab[0]} \n')
                        preds_list.append(pred[0]["type"])
                        labels_list.append(lab[0]["type"])
                        i+=1

            prec_micro, recall_micro, f1_micro = score(labels_list, preds_list, verbose=True)
            self.log('test_prec_micro', prec_micro)
            self.log('test_recall_micro', recall_micro)
            self.log('test_F1_micro', f1_micro)

        # added to accomodate trainer.predict
    def predict_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0):
        gen_kwargs = {
            "max_length": self.hparams.val_max_target_length
            if self.hparams.val_max_target_length is not None
            else self.config.max_length,
            "early_stopping": False,
            "length_penalty": 0,
            "no_repeat_ngram_size": 0,
            "num_beams": self.hparams.eval_beams if self.hparams.eval_beams is not None else self.config.num_beams,
        }
        generated_tokens = self.model.generate(
            batch["input_ids"].to(self.model.device),
            attention_mask=batch["attention_mask"].to(self.model.device),
            use_cache=True,
            **gen_kwargs,
        )
        decoded_preds = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)
        return decoded_preds

    def configure_optimizers(self):
        """
        FROM PYTORCH LIGHTNING DOCUMENTATION

        Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Return:
            Any of these 6 options.

            - Single optimizer.
            - List or Tuple - List of optimizers.
            - Two lists - The first list has multiple optimizers, the second a list of LR schedulers (or lr_dict).
            - Dictionary, with an 'optimizer' key, and (optionally) a 'lr_scheduler'
              key whose value is a single LR scheduler or lr_dict.
            - Tuple of dictionaries as described, with an optional 'frequency' key.
            - None - Fit will run without any optimizer.
        """
        print("I'M IN CONFIGURE OPTIMIZER\n")
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer_cls = Adafactor if self.hparams.adafactor else AdamW
        if self.hparams.adafactor:
            optimizer_cls = Adafactor
            optimizer_kwargs = {"scale_parameter": False, "relative_step": False}
        else:
            optimizer_cls = AdamW
            optimizer_kwargs = {
                "betas": (self.hparams.adam_beta1, self.hparams.adam_beta2),
                "eps": self.hparams.adam_epsilon,
            }

        optimizer_kwargs["lr"] = self.hparams.learning_rate

        optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        lr_scheduler = self._get_lr_scheduler(self.hparams.max_steps, optimizer)

        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]

    def _get_lr_scheduler(self, num_training_steps, optimizer):
        schedule_func = arg_to_scheduler[self.hparams.lr_scheduler]
        if self.hparams.lr_scheduler == "constant":
            scheduler = schedule_func(optimizer)
        elif self.hparams.lr_scheduler == "constant_w_warmup":
            scheduler = schedule_func(optimizer, num_warmup_steps=self.hparams.warmup_steps)
        elif self.hparams.lr_scheduler == "inverse_square_root":
            # args = {"warmup_updates": self.hparams.warmup_steps, "lr": [self.hparams.learning_rate]}
            scheduler = schedule_func(optimizer, num_warmup_steps=self.hparams.warmup_steps)
        else:
            scheduler = schedule_func(
                optimizer, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=num_training_steps
            )
        return scheduler


    def compute_metrics(self, preds, labels):
        metric_name = "rouge" # if self.hparams.task.startswith("summarization") else "sacrebleu"
        metric = load_metric(metric_name)
        def postprocess_text(preds, labels):
            preds = [pred.strip() for pred in preds]
            labels = [label.strip() for label in labels]

            # rougeLSum expects newline after each sentence
            if metric_name == "rouge":
                preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
                labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
            else:  # sacrebleu
                labels = [[label] for label in labels]

            return preds, labels
        # preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        if self.hparams.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        if metric_name == "rouge":
            result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
            # Extract a few results from ROUGE
            result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        else:
            result = metric.compute(predictions=decoded_preds, references=decoded_labels)
            result = {"bleu": result["score"]}

        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

