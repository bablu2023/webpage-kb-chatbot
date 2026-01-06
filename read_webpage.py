import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
BASE_URL = "https://en.wikipedia.org/wiki/Lithium-ion_battery"
MAX_PAGES = 10   # limit for safety

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

# --------------------------------------------------
# FUNCTIONS
# --------------------------------------------------
def clean_html(html):
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


def crawl_website(start_url, max_pages=10):
    visited = set()
    to_visit = [start_url]
    all_text = ""

    while to_visit and len(visited) < max_pages:
        url = to_visit.pop(0)
        if url in visited:
            continue

        try:
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            visited.add(url)

            page_text = clean_html(response.text)
            all_text += "\n\n" + page_text

            links = get_internal_links(start_url, response.text)
            for link in links:
                if link not in visited and link not in to_visit:
                    to_visit.append(link)

            print(f"✔ Crawled: {url}")

        except Exception as e:
            print(f"✖ Skipped {url}: {e}")

    return all_text, visited

# --------------------------------------------------
# RUN CRAWLER
# --------------------------------------------------
text, pages = crawl_website(BASE_URL, MAX_PAGES)

# --------------------------------------------------
# SAVE OUTPUT
# --------------------------------------------------
with open("webpage_text.txt", "w", encoding="utf-8") as f:
    f.write(text)

with open("pages_crawled.txt", "w", encoding="utf-8") as f:
    for p in pages:
        f.write(p + "\n")

print("\n✅ Website crawling completed")
print("📄 Text file: webpage_text.txt")
print("📄 Pages list: pages_crawled.txt")
print("🔢 Pages indexed:", len(pages))
print("🔢 Characters extracted:", len(text))
