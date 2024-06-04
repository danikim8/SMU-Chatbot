from flask import Flask, request, jsonify, render_template
import openai
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import intent_model as im
from dotenv import load_dotenv
import os


load_dotenv()  # .env 파일 로드

app = Flask(__name__)

# OpenAI API 키 설정
openai.api_key = os.getenv("OPENAI_API_KEY")# 실제 API 키로 교체


# CSV 파일 로드
# csv_files = {
#     "종합안내": pd.read_csv('종합안내.csv'),
#     "교과목강의정보": pd.read_csv('교과목강의정보.csv'),
#     "동아리": pd.read_csv('동아리.csv'),
#     "교수정보": pd.read_csv('교수정보.csv'),
#     "시설정보": pd.read_csv('시설정보.csv'),
#     "학사일정": pd.read_csv('학사일정.csv')
# }

# CSV 파일 로드
csv_data = {key: pd.read_csv(path) for key, path in im.csv_files.items()}

# csv를 문자열로 변경하는 함수
def create_context(dataframe):
    context = ""
    for i, row in dataframe.iterrows():
        row_str = " ".join(str(value) for value in row)
        context += row_str + "\n"
    return context

@app.route('/')
def index():
    return render_template('index.html')  # 템플릿 렌더링

@app.route('/get-answer', methods=['POST'])
def get_answer():
    data = request.json
    question = data.get('question', '').lower()
    

    # 학사일정 CSV 파일의 내용을 맥락으로 포함
    # faq_df = csv_files["학사일정"] # 의도를 분류하는 함수로 변경할 것 이건 임시
    
    # 질문의 의도를 분류
    intent_key = im.classify_intent(question)
    print("의도 된 분류 : " + intent_key)
    
    faq_df = csv_data[intent_key]
    context = create_context(faq_df)

    try:
        # GPT-3.5 Turbo API를 사용하여 질문에 대한 답변 얻기
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": """
                너는 상명대학교 챗봇 수뭉이야. 제공된 CSV 파일의 내용을 참고하여 질문에 대한 답변을 제공해줘.
                지켜줄 것
                - 귀여운 말투를 쓸 것
                - 반말을 쓸 것
                - 50자 이내로 답변할 것
                - 만약 csv 파일에 적당한 내용이 없다면 적당한 답을 하거나 해당 정보는 없다고 해
                - 답변할 때 csv파일에 없어서 모든다는 말을 하지 말고 자연스럽게 답변할 것
                - 이전 답변을 기반으로 답변을 계속 해줘
                """},
                {"role": "user", "content": f"질문: {question}\n\n{context}"}
            ],
            max_tokens=100
        )
        # 응답에서 텍스트 추출
        answer = response['choices'][0]['message']['content'].strip()
        print(f"Answer from OpenAI: {answer}")
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        answer = f"Error: {str(e)}"
    
    return jsonify({'answer': answer})



if __name__ == '__main__':
    app.run(debug=True)
    

# http://127.0.0.1:5000/