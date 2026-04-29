import feedparser
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from dateutil import parser as dp
from volcenginesdkarkruntime import Ark
import requests

# 北京时间 UTC+8
BJ_TZ = timezone(timedelta(hours=8))

def to_beijing_time(raw_time):
    """将各种格式的时间统一转为北京时间字符串"""
    if not raw_time:
        return datetime.now(BJ_TZ).strftime("%Y-%m-%d %H:%M:%S")
    try:
        dt = dp.parse(raw_time)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        dt = dt.astimezone(BJ_TZ)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return raw_time

def fetch_article_text(url):
    """尝试抓取文章正文，失败则返回空字符串"""
    try:
        resp = requests.get(url, timeout=10, headers={
            "User-Agent": "Mozilla/5.0 (compatible; NewsBot/1.0)"
        })
        resp.raise_for_status()
        html = resp.text
        text = re.sub(r'<[^>]+>', ' ', html)
        text = re.sub(r'\s+', ' ', text).strip()
        return text[:3000]
    except Exception:
        return ""

# ==========================================
# 1. 从配置文件加载信源
# ==========================================
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sources.json")

def load_sources():
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

SOURCES = load_sources()

API_KEY = os.environ.get("ARK_API_KEY")
MODEL_NAME = "deepseek-v3-2-251201"
client = Ark(api_key=API_KEY)

# ==========================================
# 2. AI 处理函数 (适配火山引擎 Ark API)
# ==========================================
def analyze_with_llm(title, summary, article_text="", max_retries=3):
    # 优先用正文，没有就用 RSS 摘要
    content = article_text if article_text else summary
    content = content[:1500]

    prompt = f"""你是一个资深的科技/商业主编。请完成以下两个任务：

1. 评估这篇资讯对互联网从业者、投资人或开发者的价值(0-100分)
2. 用中文写一段约100字的摘要，提炼核心信息和关键结论，让读者不打开链接也能获取核心价值

标题：{title}
内容：{content}

请严格只输出纯 JSON 格式，不要有任何 Markdown 标记：
{{"score": 评分数字, "summary": "100字左右的中文摘要"}}"""

    for attempt in range(max_retries):
        try:
            time.sleep(3.0)
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                thinking={"type": "disabled"},
            )

            content = response.choices[0].message.content
            content = content.replace("```json", "").replace("```", "").strip()
            result = json.loads(content)

            return result.get('score', 0), result.get('summary', '无摘要')

        except Exception as e:
            print(f"[-] Ark 解析报错 (尝试 {attempt+1}/{max_retries}): {e}")
            wait_time = 10 * (2 ** attempt)
            print(f"    等待 {wait_time} 秒后重试...")
            time.sleep(wait_time)

    return 0, "解析超时或失败"

# ==========================================
# 3. 主流程
# ==========================================
def main():
    if not API_KEY:
        print("致命错误：找不到 ARK_API_KEY 环境变量！")
        return

    final_data = {
        "update_time": datetime.now(BJ_TZ).strftime("%Y-%m-%d %H:%M:%S"),
        "total_scanned": 0,
        "articles": []
    }

    # 第一阶段：解析所有 RSS，收集条目
    all_entries = []
    for category, urls in SOURCES.items():
        print(f">>> 解析 RSS: {category}")
        for url in urls:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:3]:
                    summary_text = getattr(entry, 'summary', getattr(entry, 'description', '无摘要'))
                    all_entries.append({
                        "category": category,
                        "entry": entry,
                        "summary_text": summary_text,
                        "source_name": feed.feed.get('title', '未知来源')
                    })
            except Exception as e:
                print(f"[!] 解析 {url} 失败: {e}")
                continue

    final_data["total_scanned"] = len(all_entries)
    print(f"共解析到 {len(all_entries)} 篇文章，开始并发抓取正文...")

    # 第二阶段：并发抓取所有文章正文
    article_texts = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(fetch_article_text, item["entry"].link): item["entry"].link
            for item in all_entries
        }
        for future in as_completed(futures):
            link = futures[future]
            article_texts[link] = future.result()

    print("正文抓取完成，开始 AI 分析...")

    # 第三阶段：逐篇 AI 分析（受速率限制，串行调用）
    for item in all_entries:
        entry = item["entry"]
        article_text = article_texts.get(entry.link, "")

        score, summary = analyze_with_llm(entry.title, item["summary_text"], article_text)
        print(f"[{score}分] {entry.title[:30]}...")

        if score >= 85:
            final_data["articles"].append({
                "category": item["category"],
                "title": entry.title,
                "link": entry.link,
                "score": score,
                "summary": summary,
                "source_name": item["source_name"],
                "publish_time": to_beijing_time(getattr(entry, 'published', None))
            })

    final_data["articles"] = sorted(final_data["articles"], key=lambda x: x["score"], reverse=True)

    with open("feed.json", "w", encoding="utf-8") as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)

    print(f"=== 执行完毕！筛选出 {len(final_data['articles'])} 条干货。 ===")

if __name__ == "__main__":
    main()
