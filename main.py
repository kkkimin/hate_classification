

# 라이브러리 임포트

import os
import sys  # 추가된 부분
import argparse
import wandb
from model import train


# parse_args 함수(인자) : model 학습 및 추론에 쓰일 config 를 관리

def parse_args():
    """학습(train)과 추론(infer)에 사용되는 arguments를 관리하는 함수"""
    parser = argparse.ArgumentParser(description="Training and Inference arguments")

    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="/content/drive/MyDrive/git_clone",
        help="데이터셋 디렉토리 경로",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="bert",
        help='모델 타입 (예: "bert", "electra")',
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="klue/bert-base",
        help='모델 이름 (예: "klue/bert-base", "monologg/koelectra-base-finetuned-nsmc")',
    )
    parser.add_argument(
        "--save_path", type=str, default="/content/drive/MyDrive/git_clone/model", help="모델 저장 경로"
    )
    parser.add_argument(
        "--save_step", type=int, default=1000, help="모델을 저장할 스텝 간격"  # 모델이 훈련 도중에 저장되는 간격을 설정
    )
    parser.add_argument(
        "--logging_step", type=int, default=100, help="로그를 출력할 스텝 간격"
    )
    parser.add_argument(
        "--eval_step", type=int, default=1000, help="모델을 평가할 스텝 간격"
    )
    parser.add_argument(
        "--save_limit", type=int, default=7, help="저장할 모델의 최대 개수"
    )
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드 값")
    parser.add_argument("--epochs", type=int, default=10, help="에폭 수 (예: 10)")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="배치 사이즈 (메모리에 맞게 조절, 예: 16 또는 32)",
    )
    parser.add_argument(
        "--max_len", type=int, default=128, help="입력 시퀀스의 최대 길이"
    )
    parser.add_argument("--lr", type=float, default=2e-5, help="학습률(learning rate)")
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="가중치 감소(weight decay) 값"
    )
    parser.add_argument("--warmup_steps", type=int, default=2073, help="워밍업 스텝 수")
    parser.add_argument(
        "--scheduler", type=str, default="cosine", help="학습률 스케줄러 타입"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="/content/drive/MyDrive/git_clone/best_model",
        help="추론 시 불러올 모델의 경로",
    )

    # wandb 수정 이름 변경!
    parser.add_argument(
        "--run_name",
        type=str,
        default="bert-0930_5th",
        help="wandb 에 기록되는 run name",
    )

    args = parser.parse_args()
    return args

#----------------------------------------------------------------------------------------------------------------------------------------

from tqdm import tqdm

if __name__ == "__main__":
    sys.argv = sys.argv[:1]
    os.environ["TOKENIZERS_PARALLELISM"] = "false"    # tokenizer 사용 시 warning 방지
    args = parse_args()                               # 지정한 인자들을 받아와서 args 객체에 저장

    wandb.init(project="KIN", name=args.run_name)    # 프로젝트 이름을 설정하여 실험 기록 시작
    train(args)                                       # train 함수를 호출하여 실제로 모델 학습(train)을 진행함.

