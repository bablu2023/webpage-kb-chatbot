from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1️⃣ Load extracted webpage text
with open("webpage_text.txt", "r", encoding="utf-8") as f:
    text = f.read()

print("🔢 Total characters:", len(text))

# 2️⃣ Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

# 3️⃣ Split text into chunks
chunks = text_splitter.split_text(text)

print("🧩 Total chunks created:", len(chunks))

# 4️⃣ Save chunks to file (for verification)
with open("chunks.txt", "w", encoding="utf-8") as f:
    for i, chunk in enumerate(chunks):
        f.write(f"\n--- CHUNK {i+1} ---\n")
        f.write(chunk)

print("✅ Text chunking completed")
print("📄 Output file: chunks.txt")
