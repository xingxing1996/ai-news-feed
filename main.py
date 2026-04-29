import feedparser
import json
import os
import time
from datetime import datetime, timezone, timedelta
from dateutil import parser as dp
from volcenginesdkarkruntime import Ark

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

# ==========================================
# 1. 核心信源矩阵
# ==========================================
SOURCES = {
    "科技创投": [
        "https://news.ycombinator.com/rss",
        "https://techcrunch.com/feed/",
        "https://www.theverge.com/rss/index.xml",
        "https://stratechery.com/feed/"
    ],
    "商业金融": [
        "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
        "https://search.cnbc.com/rs/search/combinedcms/view.xml?profile=1000014&id=10000664",
    ],
    "科学深度": [
        "https://www.nature.com/nature.rss",
        "https://export.arxiv.org/rss/cs.AI"
    ],
    "中国视角": [
        "https://rsshub.rssforever.com/36kr/newsflashes",
        "https://rsshub.rssforever.com/latepost"
    ]
}

API_KEY = os.environ.get("ARK_API_KEY")
MODEL_NAME = "deepseek-v3-2-251201"
client = Ark(api_key=API_KEY)

# ==========================================
# 2. AI 处理函数 (适配火山引擎 Ark API)
# ==========================================
def analyze_with_llm(title, summary, max_retries=3):
    prompt = f"""你是一个资深的科技/商业主编。请评估以下资讯对互联网从业者、投资人或开发者的价值(0-100分)。
标题：{title}
摘要/内容：{summary[:500]}...

请严格只输出纯 JSON 格式，不要有任何 Markdown 标记：
{{"score": 评分数字, "insight": "一句话核心洞察(20字以内)"}}"""

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
            # 清理可能的 markdown 块包裹
            content = content.replace("```json", "").replace("```", "").strip()
            result = json.loads(content)

            return result.get('score', 0), result.get('insight', '无洞察')

        except Exception as e:
            print(f"[-] Ark 解析报错 (尝试 {attempt+1}/{max_retries}): {e}")
            # 指数退避
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

    for category, urls in SOURCES.items():
        print(f">>> 开始抓取分类: {category}")
        for url in urls:
            try:
                print(f"正在读取: {url}")
                feed = feedparser.parse(url)

                for entry in feed.entries[:3]:
                    final_data["total_scanned"] += 1
                    summary_text = getattr(entry, 'summary', getattr(entry, 'description', '无摘要'))

                    score, insight = analyze_with_llm(entry.title, summary_text)
                    print(f"[{score}分] {entry.title[:30]}...")

                    if score >= 85:
                        final_data["articles"].append({
                            "category": category,
                            "title": entry.title,
                            "link": entry.link,
                            "score": score,
                            "insight": insight,
                            "source_name": feed.feed.get('title', '未知来源'),
                            "publish_time": to_beijing_time(getattr(entry, 'published', None))
                        })
            except Exception as e:
                print(f"[!] 抓取 {url} 失败: {e}")
                continue

    final_data["articles"] = sorted(final_data["articles"], key=lambda x: x["score"], reverse=True)

    with open("feed.json", "w", encoding="utf-8") as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)

    print(f"=== 执行完毕！筛选出 {len(final_data['articles'])} 条干货。 ===")

if __name__ == "__main__":
    main()
