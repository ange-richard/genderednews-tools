import omegaconf
import hydra
import pytorch_lightning as pl

from pl_data_modules_for_inference import InferencePLDataModule
from pl_modules import BasePLModule
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer

from prediction_writer import PredictionWriter
import time


def infer(conf: omegaconf.DictConfig) -> None:
    pl.seed_everything(conf.seed)

    config = AutoConfig.from_pretrained(
        conf.config_name if conf.config_name else conf.model_name_or_path,
        decoder_start_token_id=0,
        early_stopping=False,
        no_repeat_ngram_size=0,
    )
    print("DEBUG: Loading tokenizer .......\n")
    tokenizer_kwargs = {
        "use_fast": conf.use_fast_tokenizer,
        "additional_special_tokens": ['<triplet>'],
    }
    tokenizer = AutoTokenizer.from_pretrained(
        conf.tokenizer_name if conf.tokenizer_name else conf.model_name_or_path,
        **tokenizer_kwargs
    )
    if conf.dataset_name.split('/')[-1] == 'conll04_typed.py':
        tokenizer.add_tokens(['<peop>', '<org>', '<other>', '<loc>'], special_tokens = True)
    if conf.dataset_name.split('/')[-1] == 'nyt_typed.py':
        tokenizer.add_tokens(['<loc>', '<org>', '<per>'], special_tokens = True)
    if conf.dataset_name.split('/')[-1] == 'docred_typed.py':
        tokenizer.add_tokens(['<loc>', '<misc>', '<per>', '<num>', '<time>', '<org>'], special_tokens = True)
    if conf.dataset_name.split('/')[-1] == 'rcv2.py': # add our specific entity types (names to debate)
        tokenizer.add_tokens(['<per>', '<peop>', '<pron>', '<org>', '<cue>', '<dirquot>', '<indquot>', '<mixquot>'], special_tokens = True)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        conf.model_name_or_path,
        config=config,
    )
    model.resize_token_embeddings(len(tokenizer))

    pl_module = BasePLModule(conf, config, tokenizer, model)
    print("DEBUG: Loading model ....\n")
    pl_module = pl_module.load_from_checkpoint(checkpoint_path=conf.checkpoint_path, conf=conf, config=config,
                                               tokenizer=tokenizer, model=model)

    pl_data_module = InferencePLDataModule(conf, tokenizer)
    pl_data_module.prepare_data()
    pl_data_module.setup()

    # load the data and prepare the trainer.
    paragraphs = pl_data_module.load_texts_from_jsonl(pl_data_module.conf.inference_file)
    print(f"{len(paragraphs)} input texts/paragraphs.\n")

    callbacks_store = []
    callbacks_store.append(PredictionWriter(conf.output_file, write_interval="epoch"))
    # Change accelerator argument to "gpu" if available
    trainer = pl.Trainer(devices=1, accelerator="cpu", callbacks=callbacks_store)

    print("DEBUG: Launching prediction ......\n")
    start_pred_time = time.time()
    trainer.predict(pl_module,
                    pl_data_module.inference_dataloader(paragraphs))

    print("Prediction done. %s seconds taken for prediction\n" % (time.time() - start_pred_time))


@hydra.main(config_path='../conf', config_name='root_infer')
def main(conf: omegaconf.DictConfig):
    start_time = time.time()
    infer(conf)
    print("----- %s seconds taken in total ------" % (time.time() - start_time))


if __name__ == '__main__':
    main()
