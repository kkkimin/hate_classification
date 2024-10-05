import torch.nn as nn
import torch
from transformers import BertModel, BertPreTrainedModel


# (기존)AutoModelForSequenceClassification -> (변경)CustomBERTModel  (더 많은 패턴 학습을 위해)
class CustomBERTModel(BertPreTrainedModel):
    def __init__(self, config):
        super(CustomBERTModel, self).__init__(config)
        self.bert = BertModel(config)  # BERT 모델을 가져옴
        
        # 추가 레이어 정의 (Linear 레이어와 활성화 함수 추가)
        self.additional_layer = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),  # 첫 번째 Linear 레이어
            nn.ReLU(),  # 활성화 함수
            nn.Linear(config.hidden_size, config.num_labels)  # 출력 레이어
        )
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        # BERT의 기본 출력 (sequence_output, pooled_output을 가져옴)
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # BERT의 pooled_output을 추가 레이어에 통과
        pooled_output = outputs[1]
        logits = self.additional_layer(pooled_output)  # 추가 레이어를 거친 결과
        
        return logits

