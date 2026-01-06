import os
import streamlit as st
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
import numpy as np
from openai import OpenAI
from urllib.parse import urljoin, urlparse

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Webpage RAG Chatbot (Level-4)",
    layout="centered"
)
st.title("🌐 Webpage RAG Chatbot (Level-4)")

# --------------------------------------------------
# OPENAI SETUP
# --------------------------------------------------
if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY not found. Please set it in environment variables or Streamlit Secrets.")
    st.stop()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# --------------------------------------------------
# FUNCTIONS
# --------------------------------------------------
def extract_clean_text(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    return " ".join(text.split())


def get_internal_links(base_url, html):
    soup = BeautifulSoup(html, "html.parser")
    base_domain = urlparse(base_url).netloc
    links = set()

    for tag in soup.find_all("a", href=True):
        href = tag["href"]
        full_url = urljoin(base_url, href)
        parsed = urlparse(full_url)

        if parsed.scheme in ["http", "https"] and parsed.netloc == base_domain:
            links.add(full_url.split("#")[0])

    return list(links)


def crawl_website(base_url, max_pages=10):
    visited = set()
    to_visit = [base_url]
    all_text = ""

    while to_visit and len(visited) < max_pages:
        url = to_visit.pop(0)
        if url in visited:
            continue

        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            r.raise_for_status()
            visited.add(url)

            page_text = extract_clean_text(r.text)
            all_text += " " + page_text

            links = get_internal_links(base_url, r.text)
            for link in links:
                if link not in visited and link not in to_visit:
                    to_visit.append(link)

        except Exception:
            continue

    return all_text, visited


def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def build_faiss_index(chunks):
    vectorizer = TfidfVectorizer(
        max_features=3000,
        stop_words="english"
    )
    vectors = vectorizer.fit_transform(chunks).toarray().astype("float32")

    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)

    return index, vectorizer


def generate_answer(context, question):
    prompt = f"""
Answer the question using ONLY the context below.
If the answer is not present, say "Answer not found in the website content."

Context:
{context}

Question:
{question}

Answer:
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=300
    )
    return response.choices[0].message.content.strip()

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------
if "index" not in st.session_state:
    st.session_state.index = None
    st.session_state.vectorizer = None
    st.session_state.chunks = None

# --------------------------------------------------
# TRAINING SECTION
# --------------------------------------------------
st.subheader("🔁 Train on Website (All Internal Pages)")

url = st.text_input(
    "Enter base website URL",
    placeholder="https://scml.iitm.ac.in/"
)

max_pages = st.slider("Maximum pages to crawl", 3, 20, 10)

if st.button("🚀 Train on Website"):
    if not url:
        st.warning("Please enter a valid website URL.")
    else:
        with st.spinner("Crawling website and building knowledge base..."):
            try:
                text, pages = crawl_website(url, max_pages=max_pages)
                chunks = chunk_text(text)

                index, vectorizer = build_faiss_index(chunks)

                st.session_state.index = index
                st.session_state.vectorizer = vectorizer
                st.session_state.chunks = chunks

                st.success(f"Training completed ✅")
                st.info(f"📄 Pages indexed: {len(pages)}")
                st.info(f"🧩 Chunks created: {len(chunks)}")

            except Exception as e:
                st.error(f"Training failed: {e}")

# --------------------------------------------------
# CHAT SECTION (LEVEL-4)
# --------------------------------------------------
st.subheader("💬 Ask a Question")

query = st.text_input("Type your question")

if query:
    if st.session_state.index is None:
        st.warning("Please train on a website first.")
    else:
        query_vector = (
            st.session_state.vectorizer
            .transform([query])
            .toarray()
            .astype("float32")
        )

        _, indices = st.session_state.index.search(query_vector, k=3)

        context = "\n\n".join(
            st.session_state.chunks[idx] for idx in indices[0]
        )

        with st.spinner("Generating answer..."):
            answer = generate_answer(context, query)

        st.markdown("### ✅ Final Answer")
        st.write(answer)

        with st.expander("🔍 Retrieved Context"):
            st.write(context)
