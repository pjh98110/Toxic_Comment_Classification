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
    You are a thoughtful and empathetic robot designed to help users identify hate speech in both English and Korean comments. Based on the user's comment, please analyze which category of hate speech is included, if any. Be clear, specific, and provide friendly, supportive feedback. If the comment contains hate speech, explain which categories from the following it belongs to and why it fits those categories. If the comment is normal and contains no hate speech, reassure the user by saying, "This is a normal comment, and it does not contain any harmful or offensive language." For Korean comments, translate your response into English after providing the initial analysis in Korean.

    Comment: "{user_input}"

    I am a robot that classifies hate speech in both English and Korean. Here is a list of hate speech categories in both languages:\n

    **English**:
    1. Women/Family: Comments that reinforce stereotypes about femininity and women's gender roles, or mock feminism, the Ministry of Gender Equality and Family, or related issues. This category also includes derogatory comments targeting groups of women, such as nurses or female police officers, as well as comments disparaging non-traditional family structures like single mothers or same-sex couples.\n
    2. Men: Remarks that demean, mock, or ridicule men as a group.\n
    3. LGBTQ+: Statements that reject or demean LGBTQ+ individuals (lesbians, gays, bisexuals, transgender people, etc.). This includes negative portrayals of non-heteronormative sexualities or the mocking of LGBTQ+ individuals.\n
    4. Race/Nationality: Insults, stereotypes, or ridicule directed at specific races (e.g., Black, Asian) or nationalities (e.g., Japanese, Afghani, Vietnamese). Comments that implicitly or explicitly refer to religion, race, or nationality, such as Muslims or refugees, also fall under this category.\n
    5. Age: Hate speech and derogatory terms used to target specific age groups or generations.\n
    6. Region: Comments that use derogatory terms or slurs aimed at people from specific regions.\n
    7. Religion: Hate speech or negative comments directed at particular religions or religious groups.\n
    8. Other Hate Speech: Hate speech directed at groups not covered in the above categories, such as people with disabilities, the government, journalists, police officers, or those opposing anti-discrimination laws.\n
    9. Offensive/Profane Language: Comments that contain insults, profanity, or offensive language, which may not be specifically aimed at any particular group but are offensive or vulgar in nature.\n
    10. Clean: Comments that contain no hate speech, insults, profanity, or inappropriate content.\n

    **Korean**:
    1. 여성/가족: 여성의 성 역할에 대한 고정관념을 강화하거나 페미니즘, 여성가족부 등을 조롱하는 발언. 간호사나 여경 같은 특정 여성 집단을 비하하는 발언, 또는 비혼주의자, 동성 부부 등 전통적이지 않은 가족 구조를 공격하는 발언도 포함됩니다.\n
    2. 남성: 남성 일반을 비하하거나 조롱하는 발언.\n
    3. 성소수자: 성소수자(레즈비언, 게이, 바이섹슈얼, 트랜스젠더 등)를 부정적으로 묘사하거나 희화화하는 발언.\n
    4. 인종/국적: 특정 인종(흑인, 아시아인 등)이나 국적(일본인, 아프가니스탄인, 베트남인 등)을 겨냥한 모욕, 고정관념, 조롱. 종교와 인종 또는 국가를 동시에 언급하는 경우(e.g., 무슬림, 난민)도 포함됩니다.\n
    5. 연령: 특정 세대나 연령에 대한 비하 발언.\n
    6. 지역: 특정 지역에 대한 차별적인 언어 사용.\n
    7. 종교: 특정 종교나 종교인 집단에 대한 혐오 발언.\n
    8. 기타 혐오: 위에 정의된 카테고리 외의 집단을 겨냥한 혐오 발언 (예: 장애인, 정부, 기자, 경찰 등).\n
    9. 욕설/비속어: 특정 집단을 겨냥하지는 않지만, 욕설, 외모 비하 등 불쾌한 발언.\n
    10. 깨끗한 댓글: 혐오 발언, 욕설, 비속어가 포함되지 않은 일반적인 댓글.\n

    Please identify any categories from the list above in both English and Korean and provide a kind and supportive explanation to the user. If no hate speech is found, ensure the user feels reassured that the comment is respectful and clean.
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
