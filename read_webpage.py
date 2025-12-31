import requests
from bs4 import BeautifulSoup

# 1️⃣ Webpage URL
URL = "https://en.wikipedia.org/wiki/Lithium-ion_battery"

# 2️⃣ Add headers (IMPORTANT to avoid 403)
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

# 3️⃣ Fetch webpage
response = requests.get(URL, headers=headers)
response.raise_for_status()

# 4️⃣ Parse HTML
soup = BeautifulSoup(response.text, "html.parser")

# 5️⃣ Remove scripts, styles, noscript
for tag in soup(["script", "style", "noscript"]):
    tag.decompose()

# 6️⃣ Extract visible text
text = soup.get_text(separator=" ")

# 7️⃣ Clean extra spaces
clean_text = " ".join(text.split())

# 8️⃣ Save to file
with open("webpage_text.txt", "w", encoding="utf-8") as f:
    f.write(clean_text)

print("✅ Webpage content extracted successfully")
print("📄 Output file: webpage_text.txt")
print("🔢 Characters extracted:", len(clean_text))
