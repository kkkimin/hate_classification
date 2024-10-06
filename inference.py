import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from torch.utils.data import DataLoader
import pandas as pd
import torch
import os

import numpy as np
from tqdm import tqdm
from model import load_model_for_inference
from data import prepare_dataset
from transformers import AutoTokenizer

# 데이터셋을 로드하는 함수 정의
def load_data(dataset_dir):                
    """csv file을 dataframe으로 load"""
    dataset = pd.read_csv(dataset_dir)  # csv 파일을 읽어서 데이터프레임으로 반환
    return dataset

def inference(model, tokenized_sent, device):
    """학습된(trained) 모델을 통해 결과를 추론하는 function"""
    dataloader = DataLoader(tokenized_sent, batch_size=16, shuffle=False)
    model.eval()
    output_pred = []
    for i, data in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            outputs = model(
                input_ids=data["input_ids"].to(device),
                attention_mask=data["attention_mask"].to(device),
            )
        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        result = np.argmax(logits, axis=-1)

        output_pred.append(result)
    return (np.concatenate(output_pred).tolist(),)

#----------------------------------------------------------------------------------------------------------------------------------------
def infer_and_eval(model_name,model_dir):
    """학습된 모델로 추론(infer)한 후에 예측한 결과(pred)를 평가(eval)"""
    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 1. Combined data 로드
    combined_data_path = os.path.join("/home/mean6021/hate_classification", "combined_train_with_punctuation.csv")  # combined_data의 경로 설정
    
    combined_data = load_data(combined_data_path)  # combined_data 로드

    # set model & tokenizer
    tokenizer, model = load_model_for_inference(model_name,model_dir)
    model.to(device)

    # 3. prepare_dataset 호출
    # prepare_dataset 함수에 combined_data를 인자로 전달
    _, _, hate_test_dataset, test_dataset = prepare_dataset(
        "/home/mean6021/hate_classification",  # 데이터셋 디렉토리
        tokenizer,                              # 초기화한 tokenizer
        256,                                    # max_length
        combined_data                           # combined_data 인자 추가
    )

    # predict answer
    pred_answer = inference(model, hate_test_dataset, device)  # model에서 class 추론
    pred = pred_answer[0]
    print("--- Prediction done ---")

# 크기와 출력 확인 코드 추가 ---------------------------------------------
    # Check if prediction and label sizes match
    if len(pred) != len(test_dataset["output"]):
        raise ValueError(f"Prediction length {len(pred)} does not match label length {len(test_dataset['output'])}")

    # 이진 분류에서 예측값이 0 또는 1인지 확인
    if not all(p in [0, 1] for p in pred):
        raise ValueError("Predictions contain values other than 0 and 1")
    # -------------------------------------------------------------------------

    # make csv file with predicted answer
    output = pd.DataFrame(
        {
            "id": test_dataset["id"],
            "input": test_dataset["input"],
            "output": pred,
        }
    )

    # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.
    result_path = "/home/mean6021/hate_classification/prediction"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    output.to_csv(
        os.path.join(result_path,"result.csv"), index=False
    )
    print("--- Save result ---")
    return output



#----------------------------------------------------------------------------------------------------------------------------------------

# 메인 함수 정의(추론,평가)
if __name__ == "__main__":
    model_name = "beomi/KcELECTRA-base"
    model_dir = "/home/mean6021/hate_classification/model/results/checkpoint-4500"

    infer_and_eval(model_name,model_dir)
