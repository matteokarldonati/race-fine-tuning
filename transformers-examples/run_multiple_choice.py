# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for multiple choice (Bert, Roberta, XLNet)."""

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from utils import FreeLBTrainer
from utils_multiple_choice import MultipleChoiceDataset, Split, processors

logger = logging.getLogger(__name__)


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    model_type: str = field(
        default=None, metadata={"help": "model type"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: str = field(metadata={"help": "The name of the task to train on: " + ", ".join(processors.keys())})
    data_dir: str = field(metadata={"help": "Should contain the data files for the task."})
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    reinit_pooler: bool = field(
        default=False, metadata={"help": "reinit pooler"}
    )
    reinit_layers: int = field(
        default=0,
        metadata={"help": "Number of layers to re-initialize"},
    )
    solve_coref: bool = field(
        default=False, metadata={"help": "Preprocess examples by performing coreference resolution"}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

@dataclass
class TrainingArguments(TrainingArguments):
    freelb: bool = field(
        default=False, metadata={"help": "Use FreeLB training"}
    )
    adv_lr: float = field(
        default=0
    )
    adv_steps: int = field(
        default=1, metadata={"help": "should be at least 1"}
    )
    adv_init_mag: float = field(
        default=0
    )
    norm_type: str = field(
        default='l2', metadata={"help": "choices=['l2', 'linf']"}
    )
    adv_max_norm: float = field(
        default=0, metadata={"help": "set to 0 to be unlimited"}
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    try:
        processor = processors[data_args.task_name]()
        label_list = processor.get_labels()
        num_labels = len(label_list)
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    model = AutoModelForMultipleChoice.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    if data_args.reinit_pooler:
        if model_args.model_type in ["bert", "roberta"]:
            encoder_temp = getattr(model, model_args.model_type)
            encoder_temp.pooler.dense.weight.data.normal_(mean=0.0, std=encoder_temp.config.initializer_range)
            encoder_temp.pooler.dense.bias.data.zero_()
            for p in encoder_temp.pooler.parameters():
                p.requires_grad = True
        elif model_args.model_type in ["xlnet", "bart", "electra"]:
            raise ValueError(f"{model_args.model_type} does not have a pooler at the end")
        else:
            raise NotImplementedError

    if data_args.reinit_layers > 0:
        if model_args.model_type in ["bert", "roberta", "electra"]:
            assert data_args.reinit_pooler or model_args.model_type == "electra"
            from transformers.modeling_bert import BertLayerNorm

            encoder_temp = getattr(model, model_args.model_type)
            for layer in encoder_temp.encoder.layer[-data_args.reinit_layers:]:
                for module in layer.modules():
                    if isinstance(module, (nn.Linear, nn.Embedding)):
                        module.weight.data.normal_(mean=0.0, std=encoder_temp.config.initializer_range)
                    elif isinstance(module, BertLayerNorm):
                        module.bias.data.zero_()
                        module.weight.data.fill_(1.0)
                    if isinstance(module, nn.Linear) and module.bias is not None:
                        module.bias.data.zero_()
        elif model_args.model_type == "xlnet":
            from transformers.modeling_xlnet import XLNetLayerNorm, XLNetRelativeAttention

            for layer in model.transformer.layer[-data_args.reinit_layers:]:
                for module in layer.modules():
                    if isinstance(module, (nn.Linear, nn.Embedding)):
                        module.weight.data.normal_(mean=0.0, std=model.transformer.config.initializer_range)
                        if isinstance(module, nn.Linear) and module.bias is not None:
                            module.bias.data.zero_()
                    elif isinstance(module, XLNetLayerNorm):
                        module.bias.data.zero_()
                        module.weight.data.fill_(1.0)
                    elif isinstance(module, XLNetRelativeAttention):
                        for param in [
                            module.q,
                            module.k,
                            module.v,
                            module.o,
                            module.r,
                            module.r_r_bias,
                            module.r_s_bias,
                            module.r_w_bias,
                            module.seg_embed,
                        ]:
                            param.data.normal_(mean=0.0, std=model.transformer.config.initializer_range)
        elif model_args.model_type == "bart":
            for layer in model.model.decoder.layers[-data_args.reinit_layers:]:
                for module in layer.modules():
                    model.model._init_weights(module)

        else:
            raise NotImplementedError

    train_dataset = (
        MultipleChoiceDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            task=data_args.task_name,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.train,
            solve_coref=data_args.solve_coref,
        )
        if training_args.do_train
        else None
    )
    eval_dataset = (
        MultipleChoiceDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            task=data_args.task_name,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.dev,
            solve_coref=data_args.solve_coref,
        )
        if training_args.do_eval
        else None
    )

    test_dataset = (
        MultipleChoiceDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            task=data_args.task_name,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.test,
            solve_coref=data_args.solve_coref,
        )
        if training_args.do_predict
        else None
    )

    test_dataset_high = (
        MultipleChoiceDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            task=data_args.task_name,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.test,
            solve_coref=data_args.solve_coref,
            group='high',
        )
        if training_args.do_predict
        else None
    )

    test_dataset_middle = (
        MultipleChoiceDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            task=data_args.task_name,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.test,
            solve_coref=data_args.solve_coref,
            group='middle',
        )
        if training_args.do_predict
        else None
    )

    def compute_metrics(p: EvalPrediction) -> Dict:
        preds = np.argmax(p.predictions, axis=1)
        return {"acc": simple_accuracy(preds, p.label_ids)}

    # Initialize our Trainer
    if training_args.freelb:
        trainer = FreeLBTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
        )

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()

        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        result = trainer.evaluate()

        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        if trainer.is_world_master():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in result.items():
                    logger.info("  %s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))

                results.update(result)

    if training_args.do_predict:
        predictions, label_ids, metrics = trainer.predict(test_dataset)
        predictions_high, label_ids_high, metrics_high = trainer.predict(test_dataset_high)
        predictions_middle, label_ids_middle, metrics_middle = trainer.predict(test_dataset_middle)

        predictions_file = os.path.join(training_args.output_dir, "test_predictions")
        labels_ids_file = os.path.join(training_args.output_dir, "test_labels_id")

        predictions_file_high = os.path.join(training_args.output_dir, "test_predictions_high")
        labels_ids_file_high = os.path.join(training_args.output_dir, "test_labels_id_high")

        predictions_file_middle = os.path.join(training_args.output_dir, "test_predictions_middle")
        labels_ids_file_middle = os.path.join(training_args.output_dir, "test_labels_id_middle")

        torch.save(predictions, predictions_file)
        torch.save(label_ids, labels_ids_file)

        torch.save(predictions_high, predictions_file_high)
        torch.save(label_ids_high, labels_ids_file_high)

        torch.save(predictions_middle, predictions_file_middle)
        torch.save(label_ids_middle, labels_ids_file_middle)

        examples_ids = []
        for input_feature in test_dataset.features:
            examples_ids.append(input_feature.example_id)

        examples_ids_file = os.path.join(training_args.output_dir, "examples_ids")
        torch.save(examples_ids, examples_ids_file)

        output_eval_file = os.path.join(training_args.output_dir, "test_results.txt")

        if trainer.is_world_master():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Test results *****")
                for key, value in metrics.items():
                    logger.info("  %s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))
                for key, value in metrics_high.items():
                    logger.info("  high %s = %s", key, value)
                    writer.write("high %s = %s\n" % (key, value))
                for key, value in metrics_middle.items():
                    logger.info("  middle %s = %s", key, value)
                    writer.write("middle %s = %s\n" % (key, value))

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
