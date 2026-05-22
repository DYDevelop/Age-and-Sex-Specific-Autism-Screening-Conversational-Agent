import streamlit as st
from glob import glob
import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 앱 제목 설정
st.set_page_config(page_title="Autistic Spectrum Disorders Chat Bot", page_icon="📚")

# 세션 상태 초기화
if "stage" not in st.session_state:
    st.session_state.stage = "info"  # info, survey, chat
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_info" not in st.session_state:
    st.session_state.user_info = {}
if "survey_responses" not in st.session_state:
    st.session_state.survey_responses = {}

# 현재 디렉토리에 db 폴더 생성
DB_DIR = os.path.join(os.getcwd(), "chroma_db")
os.makedirs(DB_DIR, exist_ok=True)

# 연령대 및 성별에 따른 설문 문항 (4개 그룹)
SURVEY_QUESTIONS = {
    "18-30개월-남아": [
        '당신이 아이의 이름을 부르면 아이가 당신을 쳐다 봅니까? ',
        '당신 또는 가족들 중에서 누군가가 눈에 띄게 속상해 하면 당신의 아이는 그 사람을 위로하려는(예를 들어 머리를 쓰다듬거나, 안아 주는) 모습을 보입니까?',
        '당신의 아이는 몇 개의 단어를 말할 수 있습니까??',
        '당신의 아이는 똑같은 행동(예를 들어 수도꼭지를 틀거나, 전등 수위치를 켰다 껐다 하거나, 문을 열었다 닫았다 하는 것)을 반복합니까?',
        '다른 사람이 당신 아이의 말을 쉽게 이해합니까?',
        '신기한 물건을 접했을 때 당신의 아이는 냄새를 맡거나 혀로 핥는 일이 자주 있습니까?',
        '친숙하지 않은 상황을 접했을 때 당신의 아이는 당신의 얼굴을 쳐다보면서 반응을 살핍니까?',
        '당신의 아이는 한두 개의 물체에 얼마나 오랫동안 관심을 지속합니까?', '당신은 아이와 눈을 맞추는 것이 얼마나 쉽습니까?',
        '당신의 아이는 소음에 지나치게 민감합니까?', 
        '당신의 아이는 혼자 놀 때, 물건을 일렬로 배열합니까? ',
        '당신의 아이는 당신이 보고 있는 곳을 따라서 봅니까?',
        '당신의 아이는 (예를 들어 실타래와 같은) 물건을 반복적으로 만지작거립니까?'
    ],
    "18-30개월-여아": [
        '당신의 아이는 똑같은 행동(예를 들어 수도꼭지를 틀거나, 전등 수위치를 켰다 껐다 하거나, 문을 열었다 닫았다 하는 것)을 반복합니까?',
        '당신의 아이는 몇 개의 단어를 말할 수 있습니까??',
        '당신의 아이는 (예를 들어 실타래와 같은) 물건을 반복적으로 만지작거립니까?',
        '당신 또는 가족들 중에서 누군가가 눈에 띄게 속상해 하면 당신의 아이는 그 사람을 위로하려는(예를 들어 머리를 쓰다듬거나, 안아 주는) 모습을 보입니까?',
        '당신의 아이는 (예를 들어 재미있는 광경을) 손가락으로 가리켜서 당신과 관심을 공유하려고 합니까?',
        '당신의 아이는 한두 개의 물체에 얼마나 오랫동안 관심을 지속합니까?',
        '친숙하지 않은 상황을 접했을 때 당신의 아이는 당신의 얼굴을 쳐다보면서 반응을 살핍니까?',
        '당신이 아이의 이름을 부르면 아이가 당신을 쳐다 봅니까? '
    ],
    "31-48개월-남아": [
        '또래들과 교대로 하는 상호작용을 하는 것에 느리거나 서투르다.', 
        '유머 감각이 있고 농담을 알아듣는다.',
        '(현실감을 잃지 않으면서) 상상력이 풍부하고, 흉내내기(가장하기)를 잘한다.', 
        '다른 사람들이 슬퍼하면 위로한다.',
        '다른 사람들이 무엇을 생각하거나 느끼는 지를 안다.', 
        '다른 사람과 상호작용할 때 자신감 있어 보인다.',
        '건강한 자신감이 있다.', 
        '무언가가 불공평할 때 알아챈다.', 
        '어른들에게 매달린다, 어른들에게 너무 의존적으로 보인다.',
        '더 나이 많은 어린이들 혹은 어른들이 하는 대화의 의미를 이해하지 못한다.',
        '자기가 누군가에게 너무 가까이 있거나 누군가의 공간을 침범하고 있을 때 그것을 안다.',
        '다른 사람들이 어디를 쳐다보나, 무엇을 귀 기울여 듣나 하는 것에 주의를 기울인다.',
        '다른 사람들이 자신을 이용하려고 하는 것을 인식하지 못한다.', 
        '기저귀나 속옷이 더러워지거나 젖었을 때 갈아 입길 원한다.'
    ],
    "31-48개월-여아": [
        '또래들과 교대로 하는 상호작용을 하는 것에 느리거나 서투르다.',
        '다른 사람들이 어디를 쳐다보나, 무엇을 귀 기울여 듣나 하는 것에 주의를 기울인다.',
        '사건들이 서로 어떻게 관련되어 있는지를 자기 또래의 다른 어린이들만큼 이해하지 못한다.',
        '자기가 너무 크게 말하거나 너무 시끄럽게 하고 있을 때 그것을 안다.', 
        '다른 사람과 상호작용할 때 자신감 있어 보인다.',
        '양육자와 쉽게 떨어진다.', 
        '무언가가 불공평할 때 알아챈다.', 
        '다른 사람들이 무엇을 생각하거나 느끼는 지를 안다.',
        '피검자에게 절친한 친구 또는 단짝 친구가 있습니까?', 
        '이 활동에서 저 활동으로 목적 없이 돌아다닌다.'
    ]
}

# 초기화 함수
@st.cache_resource
def initialize_qa_system():
    try:
        embeddings = OllamaEmbeddings(model="bge-m3")
        
        # 기존 벡터스토어가 있는지 확인
        if os.path.exists(DB_DIR) and os.listdir(DB_DIR):
            st.info("기존 벡터스토어를 불러오는 중...")
            vectorstore = Chroma(
                persist_directory=DB_DIR,
                embedding_function=embeddings
            )
            st.success(f"✅ 기존 벡터스토어를 성공적으로 불러왔습니다! (저장 위치: {DB_DIR})")
        else:
            # 벡터스토어가 없으면 새로 생성
            st.info("기존 벡터스토어가 없습니다. 새로 생성합니다...")
            
            folder_path = "../docs/*/*.pdf"
            pdf_files = glob(folder_path)
            
            if not pdf_files:
                st.error(f"PDF 파일을 찾을 수 없습니다: {folder_path}")
                st.stop()
            
            st.info(f"{len(pdf_files)}개의 PDF 파일을 로딩합니다...")
            
            all_docs = []
            for file in pdf_files:
                try:
                    loader = PyPDFLoader(file)
                    docs = loader.load()
                    all_docs.extend(docs)
                except Exception as e:
                    st.warning(f"파일 로딩 중 오류: {file} - {e}")
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            split_docs = text_splitter.split_documents(all_docs)
            
            st.info("임베딩을 생성하고 벡터스토어를 저장합니다...")
            vectorstore = Chroma.from_documents(
                documents=split_docs, 
                embedding=embeddings,
                persist_directory=DB_DIR
            )
            st.success(f"✅ 벡터스토어가 생성되어 저장되었습니다! (저장 위치: {DB_DIR})")
        
        retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
        model = ChatOllama(model="gpt-oss:120b")
        
        rag_prompt = ChatPromptTemplate.from_template("""
        당신은 자폐 스펙트럼 장애(ASD)에 대해 임상적 전문성과 부모·보호자 대상 상담 역량을 갖춘 한국어 상담 전문가입니다.
        부모 또는 보호자가 이해하기 쉬운 평이한 한국어로, 공감적이고 따뜻한 어조로 답변하십시오. 다만 과학적 정확성은 엄격히 유지해야 합니다.

        사용 규칙 (중요 - 절대 준수)
        1. 입력으로 주어진 검색된 문맥(<context>), 대화 이력(<history>), 설문 응답(<survey_responses>)에 포함된 정보만 사용하여 판단하십시오. 없는 사실을 지어내지 마십시오.
        2. 문맥에 근거해 답할 수 없으면 솔직하게 "정보가 부족하여 판단하기 어렵습니다"라고 답하고 추측을 피하십시오.
        3. 진단적 판단(확정적 진단명 부여)은 하지 마십시오. 오직 '위험도 수준(스크리닝 관점)'과 '권장 행동'을 제시하십시오. 진단은 반드시 전문의의 영역임을 명시해야 합니다.
        4. 응답은 반드시 한국어로 하십시오.

        목표 태스크
        - 제시된 정보를 바탕으로 아동의 ASD **위험도를 '상', '중', '하' 중 하나**로 분류하십시오.
        - 위험도 기준:
        * **상(高)**: 사회적 상호작용의 결핍과 반복적 행동 등 다수의 핵심 증상이 명확히 보고됨. 즉시 전문가 평가 필요.
        * **중(中)**: 일부 우려되는 특성이 있으나 모호하거나, 환경적 요인/단순 발달 지연 등 다른 가능성이 혼재됨. 전문가 상담 권장.
        * **하(低)**: 보고된 특성이 발달 연령에 적절하거나 매우 경미함. 경과 관찰 권장.

        - **[핵심 수정] 판단 근거 구체화:**
        * 단순히 "사회성이 부족합니다"라고 하지 마십시오.
        * 반드시 **설문 응답이나 문맥에 있는 구체적인 행동(예: "호명 반응이 전혀 없다", "자동차 바퀴만 굴린다")을 직접 인용**하여 근거로 드십시오.
        * 해당 행동이 ASD의 핵심 증상(사회적 의사소통 결핍 vs. 제한적/반복적 행동) 중 어디에 해당하는지 연결하여 설명하십시오.

        - 부모가 취할 **우선순위 행동(3단계)**을 명확히 제시하십시오.
        - '위험도 상'일 경우: 구체적인 진단 도구(ADOS-2, ADI-R 등)나 전문기관 방문의 필요성을 강조하십시오.
        - '위험도 하'일 경우: 안심시키되, 놓치지 말아야 할 관찰 포인트와 가정 내 자극 방법을 제시하십시오.

        출력 형식 (반드시 이 템플릿을 따르세요)
        ------------------------------------------------------------
        1) 자폐 위험도: **상 / 중 / 하** (셋 중 하나 선택)

        2) 상세 판단 근거:
        * **사회적 의사소통 영역:**
            - (입력 데이터에서 발견된 구체적 행동 인용) → (이것이 왜 우려되는지 임상적 의미 설명)
            - (예: "눈 맞춤이 거의 없다는 응답" → 비언어적 소통의 지연을 시사함)
        * **제한적/반복적 행동 및 관심사:**
            - (입력 데이터에서 발견된 구체적 행동 인용) → (임상적 의미 설명)
            - (관찰되지 않았다면 "특이사항 없음"으로 기재)
        * **기타 발달/감각 이슈:**
            - (언어 지연, 감각 과민/둔감 등 기타 특이사항)

        3) 추천 우선 행동(우선순위 높은 순, 최대 3개):
        - 1) [긴급도/행동명] — (구체적 실행 방법 및 이유. 예: 병원 예약 시 어떤 증상을 주로 말해야 하는지 팁 제공)
        - 2) ...
        - 3) ...

        4) 가정에서의 구체적 관찰 포인트 및 실천:
        - **관찰할 점:** (모호한 부분을 명확히 하기 위해 부모가 지켜봐야 할 구체적 상황 2~3가지)
        - **실천 팁:** (일상 루틴, 놀이 등 가정에서 바로 적용 가능한 중재법 2가지)

        5) 불확실성 및 한계:
        - (데이터 부족으로 판단이 어려운 영역이나, 직접 관찰하지 못한 한계점 1-2문장)

        6) 마무리(따뜻한 공감 및 격려):
        - (부모의 불안을 낮추고 행동을 독려하는 한 문장)

        ------------------------------------------------------------

        # 추가 지침 (Follow-up 대응용)
        - 사용자가 **첫 질문**을 할 때만 위 **출력 형식(템플릿)**을 엄격히 준수합니다.
        - **후속 질문**에 대해서는 템플릿을 버리고, **자연스러운 대화체**로 응답하십시오. 단, 이전에 제시한 위험도 판단과 근거의 일관성은 유지해야 합니다.
        - 새로운 정보가 제공되면 그에 맞춰 근거를 업데이트하여 설명해주십시오.

        <context>
        {context}
        </context>

        <conversation_history>
        {history}
        </conversation_history>

        <user_information>
        {user_info}
        </user_information>

        <survey_responses>
        {survey_data}
        </survey_responses>

        사용자 질문: {question}
        """)
        
        def answer_question(question, history, user_info, survey_data):
            context = retriever.invoke(question)
            formatted_context = "\n\n".join(doc.page_content for doc in context)
            
            response = model.invoke(
                rag_prompt.format(
                    context=formatted_context,
                    history=history,
                    user_info=user_info,
                    survey_data=survey_data,
                    question=question
                )
            )
            
            return StrOutputParser().invoke(response)
        
        return answer_question
        
    except Exception as e:
        import traceback
        st.error(f"초기화 중 오류가 발생했습니다: {e}")
        st.code(traceback.format_exc())
        return None

# 메인 UI
st.title("📚 Autistic Spectrum Disorders Chat Bot")

# 1단계: 신상정보 입력
if st.session_state.stage == "info":
    st.markdown("## 👶 피검사자 정보 입력")
    st.markdown("아동에 대한 기본 정보를 입력해주세요.")
    
    with st.form("user_info_form"):
        child_name = st.text_input("아동 이름 (선택사항)", placeholder="예: 홍길동")
        age_group = st.selectbox("연령대 선택", ["18-30개월", "31-48개월"])
        gender = st.radio("성별", ["남아", "여아"])
        additional_info = st.text_area("추가 정보 (선택사항)", 
                                       placeholder="예: 특이사항, 발달 이력 등")
        
        submitted = st.form_submit_button("다음 단계로 →")
        
        if submitted:
            st.session_state.user_info = {
                "이름": child_name if child_name else "미입력",
                "연령대": age_group,
                "성별": gender,
                "추가정보": additional_info if additional_info else "없음"
            }
            st.session_state.stage = "survey"
            st.rerun()

# 2단계: 설문조사
elif st.session_state.stage == "survey":
    st.markdown("## 📋 선별 설문조사")
    st.markdown(f"**{st.session_state.user_info['이름']}** ({st.session_state.user_info['연령대']}, {st.session_state.user_info['성별']})")
    st.markdown("---")
    st.markdown("아래 질문에 답변해주세요. 각 질문에 대해 '예', '아니오', '잘 모르겠음' 중 선택하고, 필요시 세부 내용을 입력하세요.")
    
    age_group = st.session_state.user_info["연령대"]
    gender = st.session_state.user_info["성별"]
    group_key = f"{age_group}-{gender}"
    questions = SURVEY_QUESTIONS[group_key]
    
    with st.form("survey_form"):
        responses = {}
        
        for i, question in enumerate(questions, 1):
            st.markdown(f"**{i}. {question}**")
            col1, col2 = st.columns([1, 2])
            
            with col1:
                answer = st.radio(
                    f"답변 {i}",
                    ["예", "아니오", "잘 모르겠음"],
                    key=f"q{i}_answer",
                    label_visibility="collapsed"
                )
            
            with col2:
                detail = st.text_input(
                    f"세부 사항 {i}",
                    placeholder="구체적인 상황이나 예시를 입력하세요 (선택사항)",
                    key=f"q{i}_detail",
                    label_visibility="collapsed"
                )
            
            responses[question] = {
                "답변": answer,
                "세부사항": detail if detail else "없음"
            }
            
            st.markdown("---")
        
        submitted = st.form_submit_button("설문 완료 및 상담 시작 →")
        
        if submitted:
            st.session_state.survey_responses = responses
            st.session_state.stage = "chat"
            
            # 초기 분석 메시지 자동 생성
            survey_summary = "초기 설문조사 결과를 바탕으로 종합적인 분석을 제공해주세요."
            st.session_state.messages.append({"role": "user", "content": survey_summary})
            st.rerun()

# 3단계: 챗봇 상담
elif st.session_state.stage == "chat":
    st.markdown("## 💬 전문가 상담")
    
    # 사이드바에 정보 표시
    with st.sidebar:
        st.markdown("### 👶 피검사자 정보")
        st.markdown(f"**이름:** {st.session_state.user_info['이름']}")
        st.markdown(f"**연령대:** {st.session_state.user_info['연령대']}")
        st.markdown(f"**성별:** {st.session_state.user_info['성별']}")
        if st.session_state.user_info['추가정보'] != "없음":
            st.markdown(f"**추가정보:** {st.session_state.user_info['추가정보']}")
        
        st.markdown("---")
        
        with st.expander("📋 설문조사 결과 보기"):
            for question, response in st.session_state.survey_responses.items():
                st.markdown(f"**Q:** {question}")
                st.markdown(f"**A:** {response['답변']}")
                if response['세부사항'] != "없음":
                    st.markdown(f"*세부사항: {response['세부사항']}*")
                st.markdown("---")
        
        st.markdown("---")
        
        if st.button("🔄 처음부터 다시 시작"):
            st.session_state.stage = "info"
            st.session_state.messages = []
            st.session_state.user_info = {}
            st.session_state.survey_responses = {}
            st.rerun()
        
        if st.button("🧹 대화 내용만 지우기"):
            st.session_state.messages = []
            # 초기 분석 메시지 다시 추가
            survey_summary = "초기 설문조사 결과를 바탕으로 종합적인 분석을 제공해주세요."
            st.session_state.messages.append({"role": "user", "content": survey_summary})
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 모델 정보")
        st.markdown("- 사용 모델: `gpt-oss:120b`")
        st.markdown("- 임베딩: `bge-m3`")
    
    # 시스템 초기화
    if 'qa_function' not in st.session_state:
        with st.spinner('📄 문서들을 로딩하고 벡터스토어를 생성하는 중...'):
            st.session_state.qa_function = initialize_qa_system()
            
            if st.session_state.qa_function:
                st.success("✅ 시스템이 준비되었습니다!")
            else:
                st.error("❌ 시스템 초기화에 실패했습니다.")
                st.stop()
    
    # 대화 표시 (첫 번째 메시지 제외)
    for i, message in enumerate(st.session_state.messages):
        if i == 0:  # 첫 번째 자동 생성 메시지는 표시하지 않음
            continue
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # 첫 번째 메시지 자동 처리
    if len(st.session_state.messages) == 1 and st.session_state.messages[0]["role"] == "user":
        with st.chat_message("assistant"):
            with st.spinner("설문 결과를 분석하는 중..."):
                # 사용자 정보 포맷팅
                user_info_text = f"""
                아동 이름: {st.session_state.user_info['이름']}
                연령대: {st.session_state.user_info['연령대']}
                성별: {st.session_state.user_info['성별']}
                추가 정보: {st.session_state.user_info['추가정보']}
                """
                
                # 설문 응답 포맷팅
                survey_text = ""
                for question, response in st.session_state.survey_responses.items():
                    survey_text += f"\nQ: {question}\nA: {response['답변']}"
                    if response['세부사항'] != "없음":
                        survey_text += f"\n세부사항: {response['세부사항']}"
                    survey_text += "\n"
                
                response = st.session_state.qa_function(
                    st.session_state.messages[0]["content"],
                    "",
                    user_info_text,
                    survey_text
                )
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # 사용자 입력 처리
    if prompt := st.chat_input("💬 추가 질문을 입력하세요"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # 대화 기록 텍스트 형식으로 변환
        history_text = ""
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                history_text += f"User: {msg['content']}\n"
            else:
                history_text += f"Assistant: {msg['content']}\n"
        
        # 사용자 정보 및 설문 응답 포맷팅
        user_info_text = f"""
        아동 이름: {st.session_state.user_info['이름']}
        연령대: {st.session_state.user_info['연령대']}
        성별: {st.session_state.user_info['성별']}
        추가 정보: {st.session_state.user_info['추가정보']}
        """
        
        survey_text = ""
        for question, response in st.session_state.survey_responses.items():
            survey_text += f"\nQ: {question}\nA: {response['답변']}"
            if response['세부사항'] != "없음":
                survey_text += f"\n세부사항: {response['세부사항']}"
            survey_text += "\n"
        
        with st.chat_message("assistant"):
            with st.spinner("생각 중..."):
                response = st.session_state.qa_function(
                    prompt,
                    history_text,
                    user_info_text,
                    survey_text
                )
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})