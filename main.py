import streamlit as st
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
from selenium.webdriver.chrome.options import Options
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.docstore.document import Document
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_community.vectorstores import FAISS
from langchain.schema.runnable import RunnableMap, RunnablePassthrough
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import json
import re
from langchain_teddynote import logging

load_dotenv()
logging.langsmith("Sales_Agent-fredel2")
st.title("SalesAgent")
llm = ChatOpenAI(model_name="gpt-4o")

# Selenium 설정
options = Options()
options.add_argument("--headless")  # 헤드리스 모드
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")  # 권한 문제 해결
options.add_argument("--disable-dev-shm-usage")  # 메모리 문제 해결
options.add_argument("window-size=1920x1080")
# 초기화: 단계 상태 관리
if "stage" not in st.session_state:
    st.session_state["stage"] = "URL_INPUT"

# Step 1: URL 입력 및 크롤링
if st.session_state["stage"] == "URL_INPUT":
    base_url = st.text_input("URL을 입력하세요", "")
    if st.button("크롤링 및 데이터 처리 시작"):
        # 크롤링 시작 상태로 변경
        st.session_state["stage"] = "CRAWLING"
        st.session_state["base_url"] = base_url  # URL 저장

# Step 2: 크롤링 진행 및 데이터 처리
if st.session_state["stage"] == "CRAWLING":
    with st.spinner("크롤링 중..."):
        # Selenium 설정
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")
        options.add_argument("window-size=1920x1080")
        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()), options=options
        )

        # 크롤링 진행
        base_url = st.session_state["base_url"]
        page_number = 1
        previous_content = None
        page_text = []
        while True:
            url = base_url + ("&" if "?" in base_url else "?") + f"page={page_number}"
            driver.get(url)
            time.sleep(5)
            current_content = driver.find_element(By.TAG_NAME, "body").text
            if current_content == previous_content:
                break
            previous_content = current_content
            page_text.append(current_content)
            page_number += 1
        driver.quit()

        # 데이터 처리 완료 및 단계 변경
        st.success("크롤링 완료! 데이터 추출 중...")
        st.session_state["page_text"] = page_text
        st.session_state["stage"] = "DATA_PROCESSING"

# Step 3: 데이터 처리 및 질문 입력 준비
if st.session_state["stage"] == "DATA_PROCESSING":
    prompt = f"""
    다음 텍스트에서 상품명과 가격을 추출하여 JSON 형태로 반환해주세요.

    텍스트:
    {st.session_state["page_text"]}

    출력 예시:
    [
        {{"상품명": "상품 A", "가격": 10000}},
        {{"상품명": "상품 B", "가격": 20000}}
    ]
    """

    response = llm.generate([prompt])
    response_text = response.generations[0][0].text.strip()
    if response_text.startswith("```"):
        response_text = re.sub(r"^```[^\n]*\n", "", response_text)
        response_text = re.sub(r"\n```$", "", response_text)

    try:
        extracted_data = json.loads(response_text)
        st.session_state["extracted_data"] = extracted_data
        st.session_state["stage"] = "QUESTION_INPUT"  # 질문 입력 단계로 전환
    except json.JSONDecodeError as e:
        st.error(f"JSON 파싱 오류: {e}")

# Step 4: 사용자 질문 입력 및 응답 생성
if st.session_state["stage"] == "QUESTION_INPUT":
    # 사용자 질문 입력 필드
    user_question = st.text_input("질문을 입력하세요", key="user_question")
    if st.button("입력"):
        # 질문 입력에 대한 응답 생성
        extracted_data = st.session_state["extracted_data"]

        # 가격 데이터 정규화
        prices = np.array([item["가격"] for item in extracted_data]).reshape(-1, 1)
        scaler = MinMaxScaler()
        normalized_prices = scaler.fit_transform(prices).flatten()

        # 문서 생성
        documents = [
            Document(
                page_content=f"{item['상품명']} 정규화된 가격: {normalized_price}, 실제 가격: {item['가격']}원",
                metadata={"price": item["가격"]},
            )
            for item, normalized_price in zip(extracted_data, normalized_prices)
        ]

        # 벡터 스토어 및 리트리버 생성
        embedding_model = OpenAIEmbeddings()
        faiss_vectorstore = FAISS.from_documents(
            documents=documents, embedding=embedding_model
        )
        k = 3
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = k
        faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": k})

        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5]
        )

        # ChatPromptTemplate 생성
        prompt_file_path = "./prompt/sellesAgentPrompt.txt"
        with open(prompt_file_path, "r", encoding="utf-8") as f:
            prompt_text = f.read()

        system_message = SystemMessagePromptTemplate.from_template(prompt_text)
        human_message = HumanMessagePromptTemplate.from_template("{question}")
        prompt_template = ChatPromptTemplate.from_messages(
            [system_message, human_message]
        )

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            RunnableMap(
                {
                    "context": ensemble_retriever | format_docs,
                    "question": RunnablePassthrough(),
                }
            )
            | prompt_template
            | llm
            | StrOutputParser()
        )

        # 사용자의 질문에 응답 생성
        response = rag_chain.invoke(user_question)
        st.write("응답:", response)
