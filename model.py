# 라이브러리 임포트
import pytorch_lightning as pl
import torch
from utils import compute_metrics
from data import prepare_dataset

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
)
from transformers import Trainer, TrainingArguments
from transformers import EarlyStoppingCallback
from transformers.optimization import get_cosine_with_hard_restarts_schedule_with_warmup
from tqdm import tqdm

# 수정 후
from transformers import get_cosine_schedule_with_warmup  # 추가해야 함


def load_tokenizer_and_model_for_train(args):
    """학습(train)을 위해 '사전학습(protrainde)된 토크나이저와 모델을 huggingface에서 load"""
    # load model and tokenizer
    MODEL_NAME = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # setting model hyperparameter
    model_config = AutoConfig.from_pretrained(MODEL_NAME)
    model_config.num_labels = 2
    print(model_config)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, config=model_config
    )
    print("--- Modeling Done ---")
    return tokenizer, model

#----------------------------------------------------------------------------------------------------------------------------------------
def load_model_for_inference(model_name,model_dir):
    """추론(infer)에 필요한 모델과 토크나이저 load """
    # load tokenizer
    Tokenizer_NAME = model_name
    tokenizer = AutoTokenizer.from_pretrained(Tokenizer_NAME)

    ## load my model
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    return tokenizer, model
#----------------------------------------------------------------------------------------------------------------------------------------

def load_trainer_for_train(args, model, hate_train_dataset, hate_valid_dataset):
    """학습(train)을 위한 huggingface trainer 설정"""
    training_args = TrainingArguments(
        output_dir = args.save_path + "/home/mean6021/hate_classification/results",  # 모델 훈련 후 결과 파일이 저장될 경로를 설정
        save_total_limit = args.save_limit,  # 저장할 모델의 최대 개수를 설정(for 공간절약)
        save_steps = args.save_step,  # 모델을 저장할 스텝 간격을 설정
        num_train_epochs = args.epochs,  # total number of training epochs
        learning_rate = args.lr,  # learning_rate
        per_device_train_batch_size = args.batch_size,  # 각 장치에서 훈련할 배치 사이즈를 설정
        per_device_eval_batch_size = 8,  # 평가 시 사용할 배치 사이즈를 설정
        warmup_steps = args.warmup_steps,  # number of warmup steps for learning rate scheduler
        weight_decay = args.weight_decay,  # 가중치 감소 값을 설정(과적합 방지를 위해)
        logging_dir = args.save_path + "logs",  # directory for storing logs
        logging_steps = args.logging_step,  # 로그 파일이 저장될 디렉토리 경로를 설정
        eval_strategy = "steps",  # 로그를 저장할 스텝 간격을 설정
        # `no`: No evaluation during training.
        # `steps`: Evaluate every `eval_steps`.
        # `epoch`: Evaluate every end of epoch.
        eval_steps = args.eval_step,  # 평가할 스텝 간격을 설정
        load_best_model_at_end = True,  # 훈련 종료 후 최상의 모델을 로드할지 여부를 설정
        report_to = "wandb",  # W&B 로깅 활성화(로깅 툴을 지정)
        run_name = args.run_name,  # W&B에 기록될 실행 이름을 설정(args.run_name: 실험을 식별할 수 있는 이름임)
    )

    ## Add callback & optimizer & scheduler
    MyCallback = EarlyStoppingCallback(
        early_stopping_patience=6,      # 몇 에폭 동안 검증 손실이 개선되지 않으면 훈련을 중단할지를 결정
        early_stopping_threshold=0.001  # 개선을 고려하는 손실 변화의 최소 크기
    )
    # 수정 후
    from transformers import get_cosine_schedule_with_warmup  # 추가해야 함

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.weight_decay,
        amsgrad=False,
    )
    print("--- Set training arguments Done ---")

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=hate_train_dataset,  # training dataset
        eval_dataset=hate_valid_dataset,  # evaluation dataset
        compute_metrics=compute_metrics,  # define metrics function
        callbacks=[MyCallback],
        optimizers=(
            optimizer,
            get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=len(hate_train_dataset) * args.epochs,
            ),
        ),
    )
    print("--- Set Trainer Done ---")

    return trainer

#----------------------------------------------------------------------------------------------------------------------------------------
def train(args, combined_data):
    """모델을 학습(train)하고 best model을 저장"""
    # fix a seed
    pl.seed_everything(seed=42, workers=False)

    # set device
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # set model and tokenizer
    tokenizer, model = load_tokenizer_and_model_for_train(args)
    model.to(device)

    # set data
    hate_train_dataset, hate_valid_dataset, hate_test_dataset, test_dataset = (
        prepare_dataset(args.dataset_dir, tokenizer, args.max_len)
    )

    # set trainer
    trainer = load_trainer_for_train(
        args, model, hate_train_dataset, hate_valid_dataset
    )

    # train model
    print("--- Start train ---")
    trainer.train()
    print("--- Finish train ---")
    model.save_pretrained(args.model_dir)

