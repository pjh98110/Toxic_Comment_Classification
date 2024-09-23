import streamlit as st
import google.generativeai as genai
from streamlit_chat import message
import os
import requests
from streamlit_extras.colored_header import colored_header
import pandas as pd
from datetime import datetime, timedelta
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

# 페이지 구성 설정
st.set_page_config(layout="wide")

login(token=st.secrets["secrets"]["Gemma2_Token"])

if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = st.secrets["secrets"]["GEMINI_API_KEY"]

selected_chatbot = st.sidebar.selectbox(
    "원하는 LLM 모델(Gemma2, Gemini)을 선택하세요.",
    options=["Toxic Comment Classification Using the Gemini Model", "Toxic Comment Classification Using the Fine-Tuned Gemma2 Model"],
    help="선택한 LLM 모델에 따라 다른 결과를 제공합니다."
)

def load_gemma2_model():
    if "gemma2_tokenizer" not in st.session_state:
        st.session_state.gemma2_tokenizer = AutoTokenizer.from_pretrained("WPR-GMB/gemma2-2b-it-finetuned-ko-bias-detection_merged")
    if "gemma2_model" not in st.session_state:
        st.session_state.gemma2_model = AutoModelForCausalLM.from_pretrained("WPR-GMB/gemma2-2b-it-finetuned-ko-bias-detection_merged")

def gemini_prompt(user_input):
    base_prompt = f"""
    You are an AI designed to classify hate speech in English and Korean. Based on the user's comment, analyze which hate speech category (if any) it belongs to. If the comment contains hate speech, list the categories below. If the comment is normal, respond with "This is a normal comment" without listing any categories.

    User Comment: "{user_input}"

    Hate speech categories in both languages:

    **English**:
    1. Women/Family: Stereotypes about women, mockery of feminism or non-traditional families.
    2. Men: Mocking or demeaning men.
    3. LGBTQ+: Negative comments about LGBTQ+ individuals.
    4. Race/Nationality: Insults based on race or nationality (e.g., Black, Asian, Muslim, refugees).
    5. Age: Derogatory terms about specific age groups.
    6. Region: Slurs targeting specific regions.
    7. Religion: Negative comments about religious groups.
    8. Other Hate Speech: Hate directed at other groups (e.g., disabilities, police).
    9. Offensive/Profane Language: General insults, offensive language.
    10. Clean: No hate speech or offensive content.

    **Korean**:
    1. 여성/가족: 여성 고정관념, 페미니즘이나 전통적이지 않은 가족을 조롱.
    2. 남성: 남성 비하 발언.
    3. 성소수자: 성소수자에 대한 부정적 발언.
    4. 인종/국적: 인종 또는 국적에 대한 모욕 (흑인, 아시아인, 무슬림, 난민 등).
    5. 연령: 특정 세대나 나이에 대한 비하.
    6. 지역: 특정 지역에 대한 비하.
    7. 종교: 특정 종교에 대한 부정적 발언.
    8. 기타 혐오: 장애인, 경찰 등 다른 집단을 대상으로 한 혐오.
    9. 욕설/비속어: 일반적인 욕설, 비하 발언.
    10. 깨끗한 댓글: 혐오나 욕설이 없는 일반 댓글.

    Please identify the relevant categories from the list above and explain to the user.
    """
    return base_prompt


# 스트림 표시 함수
def stream_display(response, placeholder):
    text = ''
    for chunk in response:
        if parts := chunk.parts:
            if parts_text := parts[0].text:
                text += parts_text
                placeholder.write(text + "▌")
    return text

# Initialize chat history
if "gemma2_messages" not in st.session_state:
    st.session_state.gemma2_messages = [
        {"role": "system", "content": "Gemma2 파인튜닝 모델을 사용하여 악성 댓글 여부를 알려드립니다."}
    ]

if "gemini_messages" not in st.session_state:
    st.session_state.gemini_messages = [
        {"role": "model", "parts": [{"text": "Gemini를 사용하여 악성 댓글 여부를 알려드립니다."}]}
    ]


if selected_chatbot == "Toxic Comment Classification Using the Fine-Tuned Gemma2 Model":
    colored_header(
        label='Toxic Comment Classification Using the Fine-Tuned Gemma2 Model',
        description=None,
        color_name="gray-70",
    )

    # 대화 초기화 버튼
    def on_clear_chat_gpt():
        st.session_state.gemma2_messages = [
            {"role": "system", "content": "Gemma2 파인튜닝 모델을 사용하여 악성 댓글 여부를 알려드립니다."}
        ]

    st.button("대화 초기화", on_click=on_clear_chat_gpt)

    # 이전 메시지 표시
    if "gemma2_messages" not in st.session_state:
        st.session_state.gemma2_messages = [
            {"role": "system", "content": "Gemma2 파인튜닝 모델을 사용하여 악성 댓글 여부를 알려드립니다."}
        ]
        
    for msg in st.session_state.gemma2_messages:
        role = 'user' if msg['role'] == 'user' else 'assistant'
        with st.chat_message(role):
            st.write(msg['content'])

    # 사용자 입력 처리
    if prompt := st.chat_input("챗봇과 대화하기:"):
        # 모델 및 토크나이저 로드
        load_gemma2_model()
        # 사용자 메시지 추가
        st.session_state.gemma2_messages.append({"role": "user", "content": prompt})
        with st.chat_message('user'):
            st.write(prompt)

        # 모델 입력 준비
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        st.session_state.gemma2_model.to(device)
        # model.to(device)

        # 입력 토큰화 및 장치 이동
        # inputs = tokenizer(enhanced_prompt, return_tensors="pt").to(device)
        inputs = st.session_state.gemma2_tokenizer("You are a robot that classifies hate speech. Based on the user's comment, please analyze which category of hate speech is included.\n" 
                            + "Comment: " + prompt + "\n" + "I am a robot that classifies hate speech. Here is a list of hate speech categories:\n"
                            + "1. Women/Family\n2. Men\n3. LGBTQ+\n4. Race/Nationality\n5. Age\n6. Region\n7. Religion\n8. Other Hate Speech\n\n"
                            + "The categories of hate speech included in the comment are as follows: ", return_tensors="pt").to(device)

        try:
            # 모델 예측
            output = st.session_state.gemma2_model.generate(
                **inputs,
                max_new_tokens=100, 
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                eos_token_id=st.session_state.gemma2_tokenizer.eos_token_id,
                pad_token_id=st.session_state.gemma2_tokenizer.eos_token_id 
            )


            # 응답 디코딩
            generated_text = st.session_state.gemma2_tokenizer.decode(output[0], skip_special_tokens=True)

            # 응답 메시지 표시 및 저장
            st.session_state.gemma2_messages.append({"role": "assistant", "content": generated_text})
            with st.chat_message("assistant"):
                st.write(generated_text)

        except Exception as e:
            st.error(f"Gemma2 모델 예측 중 오류가 발생했습니다: {str(e)}")


elif selected_chatbot == "Toxic Comment Classification Using the Gemini Model":
    colored_header(
        label='Toxic Comment Classification Using the Gemini Model',
        description=None,
        color_name="gray-70",
    )

    # 사이드바에서 모델의 파라미터 설정
    with st.sidebar:
        st.header("모델 설정")
        model_name = st.selectbox(
            "모델 선택",
            ['gemini-1.5-flash', "gemini-1.5-pro"]
        )
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, help="생성 결과의 다양성을 조절합니다.")
        max_output_tokens = st.number_input("Max Tokens", min_value=1, value=4096, help="생성되는 텍스트의 최대 길이를 제한합니다.")
        top_k = st.slider("Top K", min_value=1, value=40, help="다음 단어를 선택할 때 고려할 후보 단어의 최대 개수를 설정합니다.")
        top_p = st.slider("Top P", min_value=0.0, max_value=1.0, value=0.95, help="다음 단어를 선택할 때 고려할 후보 단어의 누적 확률을 설정합니다.")

    st.button("대화 초기화", on_click=lambda: st.session_state.update({
        "gemini_messages": [{"role": "model", "parts": [{"text": "Gemini를 사용하여 악성 댓글 여부를 알려드립니다."}]}]
    }))

    # 이전 메시지 표시
    if "gemini_messages" not in st.session_state:
        st.session_state.gemini_messages = [
            {"role": "model", "parts": [{"text": "Gemini를 사용하여 악성 댓글 여부를 알려드립니다."}]}
        ]
        
    for msg in st.session_state.gemini_messages:
        role = 'human' if msg['role'] == 'user' else 'ai'
        with st.chat_message(role):
            st.write(msg['parts'][0]['text'] if 'parts' in msg and 'text' in msg['parts'][0] else '')

    # 사용자 입력 처리
    if prompt := st.chat_input("챗봇과 대화하기:"):
        # 사용자 메시지 추가
        st.session_state.gemini_messages.append({"role": "user", "parts": [{"text": prompt}]})
        with st.chat_message('human'):
            st.write(prompt)

        # 프롬프트 엔지니어링 적용
        enhanced_prompt = gemini_prompt(prompt)

        # 모델 호출 및 응답 처리
        try:
            genai.configure(api_key=st.session_state.gemini_api_key)
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
                "top_k": top_k,
                "top_p": top_p
            }
            model = genai.GenerativeModel(model_name=model_name, generation_config=generation_config)
            chat = model.start_chat(history=st.session_state.gemini_messages)
            response = chat.send_message(enhanced_prompt, stream=True)

            with st.chat_message("ai"):
                placeholder = st.empty()
                
            text = stream_display(response, placeholder)
            if not text:
                if (content := response.parts) is not None:
                    text = "Wait for function calling response..."
                    placeholder.write(text + "▌")
                    response = chat.send_message(content, stream=True)
                    text = stream_display(response, placeholder)
            placeholder.write(text)

            # 응답 메시지 표시 및 저장
            st.session_state.gemini_messages.append({"role": "model", "parts": [{"text": text}]})
        except Exception as e:
            st.error(f"Gemini API 요청 중 오류가 발생했습니다: {str(e)}")
