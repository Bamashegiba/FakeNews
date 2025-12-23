import requests
from bs4 import BeautifulSoup
from datetime import datetime
import json
import time


# --- Основная функция --- #
import time
import json

def NewsPars():
    # Ввод сразу нескольких разделов через запятую
    section_urls = input("Введите ссылки на разделы AP News (через запятую): ").strip().split(",")
    section_urls = [url.strip() for url in section_urls if url.strip()]
    max_articles_per_section = 200  # можно увеличить

    all_articles = []

    for section_url in section_urls:
        print(f"[INFO] Сбор ссылок на статьи из раздела: {section_url}")
        links = get_article_links(section_url, max_articles=max_articles_per_section)
        print(f"[INFO] Найдено {len(links)} ссылок. Начинаем парсинг...")

        for i, url in enumerate(links, 1):
            article = parse_article(url)
            if article:
                all_articles.append(article)
            if i % 10 == 0:
                print(f"[INFO] Спарсили {i} статей из текущего раздела...")
            time.sleep(3)  # задержка между запросами, чтобы снизить вероятность 429

        print(f"[INFO] Раздел {section_url} завершен. Спарсили {len(links)} статей.")

    print(f"[INFO] Всего спарсили {len(all_articles)} статей со всех разделов.")

    # Сохраняем в JSONL
    output_file = "apnews_articles.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for article in all_articles:
            f.write(json.dumps(article, ensure_ascii=False) + "\n")

    print(f"[INFO] Статьи сохранены в {output_file}")
    print("[INFO] Парсинг завершён!")



def fetch_html(url, retries=5, backoff=10):
    """
    Получает HTML страницы с обработкой ошибки 429 (Too Many Requests)
    и повторными попытками с экспоненциальной задержкой.

    :param url: URL страницы
    :param retries: количество попыток
    :param backoff: начальная задержка в секундах
    :return: текст HTML или None при неудаче
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    }

    session = requests.Session()
    session.headers.update(headers)

    for attempt in range(1, retries + 1):
        try:
            response = session.get(url, timeout=10)
            response.raise_for_status()
            return response.text

        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                print(f"[WARN] 429 Too Many Requests for {url}. "
                      f"Attempt {attempt}/{retries}. Waiting {backoff} sec...")
                time.sleep(backoff)
                backoff *= 2  # экспоненциальная задержка
            else:
                print(f"[ERROR] HTTP error for {url}: {e}")
                return None

        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Failed to fetch {url}: {e}")
            return None

    print(f"[ERROR] Max retries exceeded for {url}")
    return None


def parse_article(url):
    html = fetch_html(url)
    if not html:
        return None

    soup = BeautifulSoup(html, "html.parser")

    # Заголовок
    title_tag = soup.find("h1")
    title = title_tag.get_text(strip=True) if title_tag else "No Title"

    # Дата публикации
    date_tag = soup.find("meta", {"property": "article:published_time"})
    if date_tag and date_tag.get("content"):
        try:
            date = datetime.fromisoformat(date_tag.get("content").replace("Z", "+00:00")).isoformat()
        except:
            date = None
    else:
        date = None

    # Основной текст статьи
    article_body = soup.find("div", class_="ArticleBody__content___2gQno")
    if article_body:
        paragraphs = article_body.find_all("p")
    else:
        paragraphs = soup.find_all("p")  # fallback на все <p>

    text = "\n".join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])

    if not text:
        return None

    return {
        "title": title,
        "text": text,
        "date": date,
        "source": "AP News",
        "label": "real"
    }

def get_article_links(section_url, max_articles=1000):
    """Собирает ссылки на новости с раздела AP News (HTML)."""
    collected_links = set()
    html = fetch_html(section_url)
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")

    # Находим ссылки внутри блока новостей
    for a in soup.find_all("a", href=True):
        href = a['href']
        # AP News статьи обычно содержат apnews.com/article/
        if "apnews.com/article/" in href:
            full_url = href if href.startswith("http") else "https://apnews.com" + href
            collected_links.add(full_url)
        if len(collected_links) >= max_articles:
            break

    print(f"[INFO] Collected {len(collected_links)} article links")
    return list(collected_links)[:max_articles]