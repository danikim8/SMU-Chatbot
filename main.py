import openai
from flask import Flask, request, jsonify, render_template
import openai
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import intent_model as im
from dotenv import load_dotenv
import os

# OpenAI API 키 설정
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_answer(question):
    try:
        # GPT-3.5 Turbo API를 사용하여 질문에 대한 답변 얻기
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # 사용할 모델 지정
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question}
            ],
            max_tokens=50  # 응답의 최대 길이 조정
        )
        # 응답에서 텍스트 추출
        return response.choices[0].message['content'].strip()
    except Exception as e:
        return str(e)

# 사용 예시
question = "파이썬이 뭐야"
print("Question:", question)
print("Answer:", get_answer(question))

