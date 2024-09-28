# korean_AU

조금이라도 써둘게요.

data.py
ㄴhate_dataset class : tokenized 입력을 받아와서 dataset class 로 반환하는 함수
ㄴload_data : csv file 로부터 읽어와서 dataframe 을 반환하는 함수
ㄴconstruct_tokenized_dataset : dataframe 을 입력으로 받아와서 토크나이징한 후에 반환하는 함수
ㄴprepare_dataset : csv file 로부터 읽어온 데이터를 tokenized dataset 으로 반환하는 함수

model.py
ㄴload_tokenizer_and_model_for_train : huggingface 로 부터 사전학습된 tokenizer, model 을 불러와서 반환하는 함수 (이때, config.num_labels 만 2로 수정)
ㄴload_model_for_inference : model 과 tokenizer 를 반환하는데, model 은 학습된 모델 체크포인트로 부터 불러오는 함수
ㄴload_trainer_for_train : model, dataset 을 입력으로 받아서 trainer 를 반환하는 함수
ㄴtrain : model, tokenizer, dataset 을 받아와서 trainer 에 입력으로 주고 trainer.train() 으로 학습 진행한 후에 best_model 저장하는 함수

utils.py
ㄴ compute_metrics : Trainer 에서 metric 을 계산하기 위해 쓰이는 함수

main.py
ㄴ parse_args : model 학습 및 추론에 쓰일 config 를 관리합니다.

inference.py 
ㄴ inference : 학습된(trained) 모델을 통해 결과를 추론하는 function
ㄴ infer_and_eval : 학습된 모델로 추론(infer)한 후에 예측한 결과(pred) 반환하는 함수