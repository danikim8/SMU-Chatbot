import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pickle

import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pickle

# 모델 및 토크나이저 로드
model_path = './model/saved_model_0603'  # 훈련된 모델이 저장된 디렉토리 경로
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# 저장된 모델 가중치 로드 (map_location을 사용하여 CPU로 매핑)
device = torch.device('cpu')
# model.load_state_dict(torch.load(model_path + '/custom_model_weights.pth', map_location=device))
model.to(device)
model.eval()

# 레이블 인코더 로드
with open(model_path + '/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# 모델을 평가 모드로 설정
model.eval()

# CSV 파일 로드 및 문자열로 변환
csv_files = {
    "종합안내": '종합안내.csv',
    "교과목강의정보": '교과목강의정보.csv',
    "교수정보": '교수정보.csv',
    "시설정보": '시설정보.csv',
    "학사일정": '학사일정.csv'
}

def classify_intent(question):
    inputs = tokenizer(question, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    intent_id = torch.argmax(logits, dim=1).item()
    intent_key = label_encoder.inverse_transform([intent_id])[0]
    intent_key = "학사일정"
    return intent_key

if __name__ == '__main__':
    # 테스트 예제
    test_sentences = [
        "컴공 과사 번호 알려줘",
        "회계팀 전화번호 알려줘.",
        "가족과문화어느 교수님이 수업하셔?",
        "계측및제어 전심이야?",
        "정문까지 가는 길은 안전해?",
        "파이썬 전심임?",
        "C언어 전심임?",
        "유재권 교수연구소 어디?", # 교수정보
        "도서관 어디", # 시설정보
        "수강신청 개시일", # 학사일정
        "어떤 교수가 경영 빅데이터 분석을 가르치나요?",# 교과목강의정
        "애니메이션 제작부에서는 어떤 활동을 하시나요?", # 동아리
        "윤가네에서 먹을 수 있는 메뉴는",
    ]

    for sentence in test_sentences:
        intent = classify_intent(sentence)
        print(f"The intent of '{sentence}' is '{intent}'")

    print("Label encoder classes:", label_encoder.classes_)







# 테스트 질문들
# test_questions = input("질문을 입력하세요 : ")

# print("모델 테스트 시작-------")
# for question in test_sentences:
#     intent_key = classify_intent(question)
#     print(f"The intent of '{question}' is '{intent_key}'")
# print("------모델 테스트 완료-------")

# 질문을 입력 받는다.
# 질문 의도를 분류한다.
# 질문에 대한 데이터베이스를 연결한다.
# 답변을 리턴한다.

# Lmll