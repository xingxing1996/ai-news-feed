import feedparser
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from dateutil import parser as dp
from volcenginesdkarkruntime import Ark
import requests

# 北京时间 UTC+8
BJ_TZ = timezone(timedelta(hours=8))

# 常见缩写时区映射
TZ_MAP = {
    "PST": timezone(timedelta(hours=-8)),
    "PDT": timezone(timedelta(hours=-7)),
    "EST": timezone(timedelta(hours=-5)),
    "EDT": timezone(timedelta(hours=-4)),
    "CST": timezone(timedelta(hours=-6)),
    "CDT": timezone(timedelta(hours=-5)),
    "MST": timezone(timedelta(hours=-7)),
    "MDT": timezone(timedelta(hours=-6)),
    "GMT": timezone.utc,
    "BST": timezone(timedelta(hours=1)),
    "CET": timezone(timedelta(hours=1)),
    "CEST": timezone(timedelta(hours=2)),
    "JST": timezone(timedelta(hours=9)),
    "IST": timezone(timedelta(hours=5, minutes=30)),
    "AEST": timezone(timedelta(hours=10)),
    "NZST": timezone(timedelta(hours=12)),
}

def parse_time_aware(raw_time):
    """解析时间字符串，正确处理缩写时区"""
    dt = dp.parse(raw_time, tzinfos=TZ_MAP)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt

def to_beijing_time(raw_time):
    """将各种格式的时间统一转为北京时间字符串"""
    if not raw_time:
        return datetime.now(BJ_TZ).strftime("%Y-%m-%d %H:%M:%S")
    try:
        dt = parse_time_aware(raw_time)
        dt = dt.astimezone(BJ_TZ)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return raw_time

def is_similar_title(a, b, threshold=0.6):
    """简单的标题相似度检测（基于词重叠率），不会太严格"""
    def normalize(s):
        return re.sub(r'[^\w]', '', s.lower()).split()
    words_a = normalize(a)
    words_b = normalize(b)
    if not words_a or not words_b:
        return False
    common = set(words_a) & set(words_b)
    overlap = len(common) / min(len(set(words_a)), len(set(words_b)))
    return overlap >= threshold

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
def analyze_with_llm(title, summary, source_name="未知来源", article_text="", max_retries=3):
    # 优先用正文，没有就用 RSS 摘要
    content = article_text if article_text else summary
    content = content[:1500]

    prompt = f"""你是一位资深科技财经主编，正在为互联网从业者、投资人和开发者筛选有价值的资讯。

内容可能为英文、日文或中文，但所有输出必须用中文。

请评估以下文章，完成以下任务：
1. 评分(0-100)：
   - 50-59：日常更新，小版本发布
   - 60-70：有一定参考价值的技术文章或行业动态
   - 71-80：重要技术突破、行业趋势分析、有实践价值的工程经验
   - 81-90：重大产品发布、关键架构演进、影响行业的战略变化
   - 91-100：里程碑级事件，深刻影响多个行业
2. 中文标题：将原标题翻译为简洁的中文标题
3. 中文来源：将来源名称翻译为中文
4. 约100字中文摘要：提炼核心事实、关键数据/技术点和行业影响
5. 约100字机遇分析：分析此事件带来的商业机会、创业方向、投资价值或对从业者的影响

标题：{title}
来源：{source_name}
内容：{content}

请严格只输出纯 JSON，不要任何额外文字或 Markdown 标记：
{{"score": 数字, "title_cn": "中文标题", "source_cn": "中文来源名", "summary": "中文摘要", "opportunity": "机遇与商机分析"}}"""

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

            return result.get('score', 0), result

        except Exception as e:
            print(f"[-] Ark 解析报错 (尝试 {attempt+1}/{max_retries}): {e}")
            wait_time = 10 * (2 ** attempt)
            print(f"    等待 {wait_time} 秒后重试...")
            time.sleep(wait_time)

    return 0, None

def generate_daily_insight(articles):
    """将高分文章汇总，让大模型生成今日洞察报告"""
    if not articles:
        return []

    # 取评分 >= 70 的文章用于洞察分析
    top_articles = [a for a in articles if a["score"] >= 70]
    if not top_articles:
        top_articles = articles[:10]

    # 构建文章摘要列表
    article_list = ""
    for i, a in enumerate(top_articles[:20], 1):
        article_list += f"{i}. [{a['category']}] {a['title_cn']}（{a['source_cn']}，{a['score']}分）\n   摘要：{a['summary']}\n   机遇：{a['opportunity']}\n\n"

    prompt = f"""你是一位资深科技财经分析师，请基于以下 {len(top_articles[:20])} 篇今日热门资讯，生成一份"今日洞察报告"。

要求：
- 输出 3-5 条核心洞察
- 每条洞察包含：趋势标题、详细分析（150字以内）、相关投资/创业/职业机会
- 要跨领域关联分析，发现隐藏趋势
- 所有输出用中文

今日资讯：
{article_list}

请严格只输出纯 JSON 数组，不要任何额外文字或 Markdown 标记：
[{{"title": "洞察标题", "analysis": "详细分析", "action": "建议关注的方向或机会"}}]"""

    for attempt in range(3):
        try:
            time.sleep(3.0)
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                thinking={"type": "disabled"},
            )
            content = response.choices[0].message.content
            content = content.replace("```json", "").replace("```", "").strip()
            result = json.loads(content)
            if isinstance(result, list):
                return result
        except Exception as e:
            print(f"[-] 洞察生成失败 (尝试 {attempt+1}/3): {e}")
            time.sleep(10 * (2 ** attempt))

    return []

# ==========================================
# 3. 主流程
# ==========================================
def main():
    if not API_KEY:
        print("致命错误：找不到 ARK_API_KEY 环境变量！")
        sys.exit(1)

    final_data = {
        "update_time": datetime.now(BJ_TZ).strftime("%Y-%m-%d %H:%M:%S"),
        "total_scanned": 0,
        "articles": []
    }

    # 第一阶段：解析所有 RSS，收集条目
    now = datetime.now(BJ_TZ)
    cutoff = now - timedelta(hours=48) # 只保留最近48小时内的文章
    all_entries = []
    for category, urls in SOURCES.items():
        print(f">>> 解析 RSS: {category}")
        for url in urls:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:10]:
                    # 过滤掉太旧的文章
                    raw_time = getattr(entry, 'published', None)
                    if raw_time:
                        try:
                            pub_dt = parse_time_aware(raw_time)
                            pub_dt = pub_dt.astimezone(BJ_TZ)
                            if pub_dt < cutoff:
                                continue
                        except Exception:
                            pass  # 解析不了时间的就放过
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

        score, ai_result = analyze_with_llm(entry.title, item["summary_text"], item["source_name"], article_text)
        print(f"[{score}分] {entry.title[:30]}...")

        if score >= 60 and ai_result:
            final_data["articles"].append({
                "category": item["category"],
                "title": entry.title,
                "title_cn": ai_result.get("title_cn", ""),
                "link": entry.link,
                "score": score,
                "summary": ai_result.get("summary", "无摘要"),
                "opportunity": ai_result.get("opportunity", ""),
                "source_name": item["source_name"],
                "source_cn": ai_result.get("source_cn", ""),
                "publish_time": to_beijing_time(getattr(entry, 'published', None))
            })

    # 0条数据则报错退出，不更新文件
    if len(final_data["articles"]) == 0:
        print("!!! 错误：未筛选出任何文章（0条），可能所有 AI 调用均失败，不更新 feed.json !!!")
        sys.exit(1)

    # 去重：先按分数排序，然后 URL 去重 + 标题相似度去重
    final_data["articles"] = sorted(final_data["articles"], key=lambda x: x["score"], reverse=True)
    seen_urls = set()
    seen_titles = []
    deduped = []
    for article in final_data["articles"]:
        # URL 去重
        url = article.get("link", "")
        if url in seen_urls:
            continue
        seen_urls.add(url)
        # 标题相似度去重（保留高分的那条）
        title = article.get("title_cn") or article.get("title", "")
        if any(is_similar_title(title, t) for t in seen_titles):
            print(f"    去重: {title[:30]}...")
            continue
        seen_titles.append(title)
        deduped.append(article)
    final_data["articles"] = deduped
    print(f"去重后保留 {len(deduped)} 条文章")

    # 生成今日洞察
    print("开始生成今日洞察报告...")
    insights = generate_daily_insight(deduped)
    final_data["daily_insights"] = insights
    if insights:
        print(f"生成 {len(insights)} 条洞察")
    else:
        print("洞察生成失败，跳过")

    # 备份上一份 feed.json 到 feedLastTime.json
    script_dir = os.path.dirname(os.path.abspath(__file__))
    feed_path = os.path.join(script_dir, "feed.json")
    backup_path = os.path.join(script_dir, "feedLastTime.json")
    if os.path.exists(feed_path):
        import shutil
        shutil.copy2(feed_path, backup_path)
        print("已备份 feed.json -> feedLastTime.json")

    with open(feed_path, "w", encoding="utf-8") as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)

    print(f"=== 执行完毕！筛选出 {len(final_data['articles'])} 条干货，{len(insights)} 条洞察。 ===")

if __name__ == "__main__":
    main()
