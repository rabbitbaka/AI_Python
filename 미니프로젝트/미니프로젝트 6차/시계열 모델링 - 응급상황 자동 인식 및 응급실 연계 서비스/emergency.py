
import os
import json
import requests
import xml.etree.ElementTree as ET

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import *

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset, Dataset

import openai
from openai import OpenAI

from warnings import filterwarnings
FutureWarning
filterwarnings('ignore')

path = '/content/drive/MyDrive/project6_2/' # 미리 지정
er = pd.read_csv(path + '응급실 정보.csv') # 미리 지정
#filename = 'audio2.mp3' # 사용자 input 예시
#user_location = (37.35861845,127.1150359)  # 사용자 좌표 예시

# 1. load_key(path)
# 2. predict('audio_file')
# 3. recommend_hospital()


# 0-1. load file------------------
def load_file(filepath):
    with open(filepath, 'r') as file:
        return file.readline().strip()

# 0-2. load key file------------------
def load_key(filepath):
    api_key = load_file(filepath + 'api_key.txt')
    openai.api_key = api_key

    os.environ['OPENAI_API_KEY'] = api_key
    return

# 1-1 audio2text--------------------
def audio_to_text(filename):
    audio_path = path + 'audio/'
    # OpenAI 클라이언트 생성
    client = OpenAI()
    audio_file = open(audio_path + filename, "rb")
    # 오디오 파일을 읽어서, 위스퍼를 사용한 변환
    transcript = client.audio.transcriptions.create(
        file=audio_file,
        model="whisper-1",
        language="ko",
        response_format="text",
    )
    # 결과 반환
    #transcript = '출혈이 있고 의식이 없고 호흡이 없는 상태'
    return transcript

# 1-2 text2summary------------------
def text_summary(input_text):
    # OpenAI 클라이언트 생성
    client = OpenAI()

    # 시스템 역할과 응답 형식 지정
    system_role = '''
    당신은 응급 상황 텍스트에서 핵심 내용을 요약하고, 중증도를 정확히 예측해주는 어시스턴트입니다.
    참고로 중증도 6등급은 응급상황과 전혀 관계없는 텍스트일 때 출력합니다.
    응답은 다음의 형식을 지켜주세요:
    {
        "summary": "텍스트 요약",
        "severity": "예측한 중증도 ('1등급', '2등급', '3등급', '4등급', '5등급','6등급' 중 하나)"
    }
    중증도는 응급 상황의 위험성과 시급성을 기준으로 예측하세요.
    예시  요청{
    'User_Input' : "환자는 심한 흉통을 호소하며, 호흡곤란과 식은땀이 동반됩니다. 과거 병력으로는 심근경색이 있었으며, 현재 맥박이 매우 빠릅니다."
    }
    예시 출력
    {
        "summary": "환자는 심한 흉통과 호흡곤란을 호소하며, 과거 심근경색 병력이 있습니다.",
        "severity": "1등급"
    }
    예시  요청{
    'User_Input' : "푸른 정원에 나비가 있어."
    }
    예시 출력
    {
        "summary": "푸른 정원에 나비가 있어.",
        "severity": "6등급"
    }
    '''

    # 입력데이터를 GPT-3.5-turbo에 전달하고 답변 받아오기
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": system_role
            },
            {
                "role": "user",
                "content": input_text
            }
        ]
    )

    # 응답 받기
    answer = response.choices[0].message.content

    # 응답형식을 정리하고 return
    answer_dict = json.loads(answer)

    return answer_dict['summary'], answer_dict['severity']


# 2. model prediction------------------
def predict(text, model, tokenizer):
    # 입력 문장 토크나이징
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    # 모델 예측
    with torch.no_grad():
        outputs = model(**inputs)

    # 로짓을 소프트맥스로 변환하여 확률 계산
    logits = outputs.logits
    probabilities = logits.softmax(dim=1)

    # 가장 높은 확률을 가진 클래스 선택
    pred = torch.argmax(probabilities, dim=-1).item()

    return pred, probabilities

def predict_class(filename):
    save_directory = path + "km_bert/km_bert"
    model = BertForSequenceClassification.from_pretrained(save_directory)
    tokenizer = BertTokenizer.from_pretrained(save_directory)

    # summary
    text,severity = text_summary(audio_to_text(filename))
    if severity == '6등급':
      return 6, text
    inputs = tokenizer(
        text,
        max_length=128,  # 모델이 처리할 수 있는 최대 길이 
        truncation=True, 
        padding=True,    
        return_tensors="pt"  # PyTorch 텐서 반환
    )

    # Prediction
    outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(probabilities, dim=-1).item()

    return predicted_class, text, probabilities

def info_treatment(input_text):
    # OpenAI 클라이언트 생성
    client = OpenAI()

    # 시스템 역할과 응답 형식 지정
    system_role =system_role = '''
    당신은 경미한 상태의 환자를 위해 정보를 요약하고 적절한 비응급 조치를 제안하는 도우미입니다.

    역할:
    1. 환자의 상태가 심각하지 않음을 전제로 제공된 정보를 요약합니다 (`info`).
    2. 환자의 상태를 관리하기 위한 적절한 비응급 조치 순서를 제안합니다 (`treatment`).

    응답 형식:
    {
        "info": "상황 요약",
        "treatment": [
            "1. 첫 번째로 해야 할 비응급 조치",
            "2. 두 번째로 해야 할 비응급 조치",
            "3. 세 번째로 해야 할 비응급 조치 (필요 시 추가)"
        ]
    }

    예시 입력:
    {
        'User_Input': "환자가 가벼운 두통을 호소하며, 최근 충분한 수면을 취하지 못했다고 합니다. 특별한 병력은 없습니다."
    }

    예시 출력:
    {
        "info": "환자는 가벼운 두통을 호소하며, 최근 수면 부족이 원인일 가능성이 있습니다.",
        "treatment": [
            "1. 충분히 휴식을 취할 수 있도록 안내합니다.",
            "2. 물을 충분히 섭취하도록 권장합니다.",
            "3. 증상이 지속되거나 악화되면 일반의를 방문하도록 합니다."
        ]
    }
    '''
    # 입력데이터를 GPT-3.5-turbo에 전달하고 답변 받아오기
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": system_role
            },
            {
                "role": "user",
                "content": input_text
            }
        ]
    )

    # 응답 받기
    answer = response.choices[0].message.content

    # 응답형식을 정리하고 return
    answer_dict = json.loads(answer)
    return answer_dict['info'], answer_dict['treatment']

def make_emergency_data(emergecy_df):
    emergency_locations = []

    for _, row in emergecy_df.iterrows():
        emergency_locations.append({"name": row["병원이름"], "coords": (row["위도"], row["경도"])})
    return emergency_locations

def recommend_hospitals(user_location, emergency_locations, API_KEY_ID, API_KEY, max_distance=10, max_attempt_distance=50):
    import requests
    from math import radians, sin, cos, sqrt, atan2
    # Haversine 함수
    def haversine(coord1, coord2):
        R = 6371.0  # 지구 반지름 (km)
        lat1, lon1 = radians(coord1[0]), radians(coord1[1])
        lat2, lon2 = radians(coord2[0]), radians(coord2[1])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        return R * c

    # Directions API 호출 함수
    def get_travel_time_with_fallback(start_coords, end_coords):
        url = "https://naveropenapi.apigw.ntruss.com/map-direction/v1/driving"
        options = ["trafast", "traoptimal"]  # 사용할 옵션 순서
        headers = {
            "X-NCP-APIGW-API-KEY-ID": API_KEY_ID,
            "X-NCP-APIGW-API-KEY": API_KEY
        }

        for option in options:
            params = {
                "start": f"{start_coords[1]},{start_coords[0]}",  # 경도, 위도 순서
                "goal": f"{end_coords[1]},{end_coords[0]}",
                "option": option
            }

            try:
                response = requests.get(url, headers=headers, params=params)
                response.raise_for_status()
                data = response.json()

                # 'route' 데이터 확인 및 처리
                if "route" in data and option in data["route"]:
                    summary = data["route"][option][0]["summary"]
                    duration_ms = summary["duration"]  # 소요 시간 (밀리초)
                    distance_m = summary["distance"]  # 도로 거리 (미터)
                    return duration_ms / 1000, distance_m / 1000  # 초 단위, km 단위 반환
            except requests.exceptions.RequestException as e:
                print(f"{option} 옵션 실패: {e}")
            except KeyError as e:
                print(f"API 응답에서 예상치 못한 데이터 구조: {e}")

        # 모든 옵션 실패 시
        return float("inf"), float("inf")

    # 거리 기준 확장하며 병원 검색
    current_distance = max_distance
    while current_distance <= max_attempt_distance:
        filtered_hospitals = [
            loc for loc in emergency_locations
            if haversine(user_location, loc["coords"]) <= current_distance
        ]

        if filtered_hospitals:
            results = []
            for hospital in filtered_hospitals:
                travel_time, road_distance = get_travel_time_with_fallback(user_location, hospital["coords"])
                if not (travel_time is None or travel_time == float("inf")):  # 유효한 값만 추가
                    results.append({
                        "name": hospital["name"],
                        "road_distance": road_distance,
                        "travel_time": travel_time
                    })

            if results:  # 유효한 결과가 있는 경우
                top_3_hospitals = sorted(results, key=lambda x: x["travel_time"])[:3]

                print("추천 병원:")
                for hospital in top_3_hospitals:
                    duration_in_seconds = hospital["travel_time"]  # 초 단위 값
                    hours = int(duration_in_seconds // 3600)
                    minutes = int((duration_in_seconds % 3600) // 60)
                    print(
                        f"- {hospital['name']}, 도로 거리: {hospital['road_distance']:.2f}km, "
                        f"예상 소요 시간: {hours}시간 {minutes}분"
                    )
                return  # 추천 병원 출력 후 종료
        current_distance += 10

    # 최대 거리까지 검색했지만 병원 없음
    print(f"{max_attempt_distance}km 내 유효한 병원을 찾을 수 없습니다.")
