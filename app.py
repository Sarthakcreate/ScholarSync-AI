# ai_quiz_app.py

import streamlit as st
import google.generativeai as genai
import PyPDF2
import docx
import re
import faiss
import numpy as np

# ----------------------------
# 🔑 Gemini Setup
# ----------------------------
genai.configure(api_key=st.secrets.get("GEMINI_API_KEY", "your_api_key_here"))
model = genai.GenerativeModel("gemini-1.5-flash")

# ----------------------------
# 📄 Helper Functions
# ----------------------------
def extract_text_from_pdf(uploaded_file):
    text = ""
    reader = PyPDF2.PdfReader(uploaded_file)
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def extract_text_from_docx(uploaded_file):
    doc = docx.Document(uploaded_file)
    return "\n".join([para.text for para in doc.paragraphs])

def generate_quiz(text, difficulty, num_questions=5):
    """
    Generates quiz questions using Gemini API.
    Returns a formatted string with Qs, options, answers, explanations.
    """
    prompt = f"""
    You are a helpful teacher AI. Create {num_questions} {difficulty}-level quiz questions 
    from the given study material. Each question must be **multiple-choice** with 4 options. 
    Provide the correct answer and a short explanation.

    Format strictly like this:

    Q1. Question text
    a) Option A
    b) Option B
    c) Option C
    d) Option D
    Answer: b
    Explanation: short explanation here

    Study Material:
    {text}
    """

    response = model.generate_content(prompt)
    return response.text.strip()

def parse_quiz(quiz_text):
    """
    Parse Gemini quiz output into structured format for practice mode.
    """
    questions = []
    blocks = re.split(r"Q\d+\.", quiz_text)[1:]  # split by Q1., Q2., etc.
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 6:
            continue
        q_text = lines[0].strip()
        options = [line.strip() for line in lines[1:5]]
        answer_line = [l for l in lines if l.lower().startswith("answer")]
        explanation_line = [l for l in lines if l.lower().startswith("explanation")]
        correct = answer_line[0].split(":")[1].strip() if answer_line else None
        explanation = explanation_line[0].split(":", 1)[1].strip() if explanation_line else ""
        questions.append({
            "question": q_text,
            "options": options,
            "answer": correct,
            "explanation": explanation
        })
    return questions

# ----------------------------
# 📚 RAG PDF Chat Functions
# ----------------------------
def chunk_text(text, chunk_size=800, overlap=100):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

def embed_chunks(chunks):
    embeddings = []
    for ch in chunks:
        emb = genai.embed_content(model="models/embedding-001", content=ch)
        embeddings.append(emb["embedding"])
    return np.array(embeddings).astype("float32")

def create_faiss_index(embeddings):
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    return index

def retrieve(query, chunks, index, k=3):
    q_emb = genai.embed_content(model="models/embedding-001", content=query)["embedding"]
    q_emb = np.array([q_emb]).astype("float32")
    distances, indices = index.search(q_emb, k)
    return [chunks[i] for i in indices[0]]

def rag_answer(query, chunks, index):
    retrieved = retrieve(query, chunks, index)
    context = "\n".join(retrieved)

    prompt = f"""
    You are a helpful research assistant.
    Use ONLY the context below to answer the question.
    Give a clear, detailed explanation.

    Context:
    {context}

    Question: {query}
    """
    response = model.generate_content(prompt)
    return response.text

# ----------------------------
# 🎨 Streamlit UI
# ----------------------------
st.set_page_config(page_title="AI Quiz & Research Assistant", page_icon="📘", layout="wide")

st.title("📘 AI Quiz & Research Assistant")
st.markdown("Upload your notes, textbooks, or research papers. Generate quizzes OR chat with documents.")

tab1, tab2 = st.tabs(["📝 Quiz Generator", "📚 RAG PDF Chat"])

# ----------------------------
# TAB 1 - Quiz Generator
# ----------------------------
with tab1:
    st.header("Automatic Quiz & Assignment Generator")

    uploaded_file = st.file_uploader("Upload a PDF or DOCX file", type=["pdf", "docx"], key="quiz_upload")

    difficulty = st.selectbox("Select Difficulty Level", ["Easy", "Medium", "Hard"])
    num_questions = st.slider("Number of Questions", 3, 15, 5)

    if uploaded_file:
        if uploaded_file.type == "application/pdf":
            extracted_text = extract_text_from_pdf(uploaded_file)
        else:
            extracted_text = extract_text_from_docx(uploaded_file)

        st.success("✅ File uploaded and text extracted!")

        if st.button("Generate Quiz"):
            with st.spinner("Generating quiz... ⏳"):
                quiz_text = generate_quiz(extracted_text[:3000], difficulty, num_questions)

                st.markdown("### 📑 Generated Quiz")
                st.text(quiz_text)

                st.download_button(
                    label="⬇️ Download Quiz",
                    data=quiz_text,
                    file_name="quiz.txt",
                    mime="text/plain"
                )

                # ----------------------------
                # 🎮 Practice Mode
                # ----------------------------
                st.markdown("### 🎮 Student Practice Mode")

                quiz_data = parse_quiz(quiz_text)
                score = 0
                answers = {}

                for idx, q in enumerate(quiz_data):
                    st.write(f"**Q{idx+1}. {q['question']}**")
                    choice = st.radio(
                        f"Select answer for Q{idx+1}",
                        q["options"],
                        index=None,
                        key=f"q_{idx}"
                    )
                    answers[idx] = choice

                if st.button("Submit Answers"):
                    for idx, q in enumerate(quiz_data):
                        student_ans = answers.get(idx)
                        correct_option = q["options"][ord(q["answer"].lower()) - 97] if q["answer"] else None
                        if student_ans == correct_option:
                            score += 1
                            st.success(f"✅ Q{idx+1}: Correct! {q['explanation']}")
                        else:
                            st.error(f"❌ Q{idx+1}: Wrong. Correct answer: {correct_option}. {q['explanation']}")

                    st.subheader(f"🎯 Final Score: {score}/{len(quiz_data)}")

# ----------------------------
# TAB 2 - RAG PDF Chat
# ----------------------------
with tab2:
    st.header("Research Paper / PDF Assistant")

    uploaded_pdf = st.file_uploader("Upload a PDF for RAG Chat", type=["pdf"], key="rag_upload")
    if uploaded_pdf:
        text = extract_text_from_pdf(uploaded_pdf)
        chunks = chunk_text(text)

        with st.spinner("🔍 Building knowledge index..."):
            embeddings = embed_chunks(chunks)
            index = create_faiss_index(embeddings)

        st.success("✅ Document ready! Ask questions below:")

        user_query = st.text_input("Ask a question about the paper:")
        if user_query:
            with st.spinner("🤔 Thinking..."):
                answer = rag_answer(user_query, chunks, index)
            st.markdown("### 🤖 Answer")
            st.write(answer)

