#!/bin/bash
python -m src.baseline.zero_shot_inference
python -m src.baseline.monolingual_finetuning

python -m src.transfer_learning.crosslingual_transfer
python -m src.transfer_learning.multilingual_finetuning

python -m src.parameter_efficient.adapter_finetuning
python -m src.parameter_efficient.lora_finetuning

python -m src.data_augmentation.backtranslation
python -m src.data_augmentation.noise_injection
python -m src.data_augmentation.transliteration

python -m src.ensemble.ensemble_model