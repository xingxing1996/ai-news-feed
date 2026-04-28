import feedparser
import requests
import json
import os
import time
from datetime import datetime

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
        "http://export.arxiv.org/rss/cs.AI"
    ],
    "中国视角": [
        "https://rsshub.app/36kr/newsflashes",
        "https://rsshub.app/latepost"
    ]
}

API_KEY = os.environ.get("LLM_API_KEY")
# 修正为和你提供的一模一样的 URL
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent"

# ==========================================
# 2. AI 处理函数 (完全适配你提供的 curl 格式)
# ==========================================
def analyze_with_llm(title, summary, max_retries=2):
    prompt = f"""
    你是一个资深的科技/商业主编。请评估以下资讯对互联网从业者、投资人或开发者的价值(0-100分)。
    标题：{title}
    摘要/内容：{summary[:500]}...
    
    请严格只输出纯 JSON 格式，不要有任何 Markdown 标记：
    {{"score": 评分数字, "insight": "一句话核心洞察(20字以内)"}}
    """
    
    # 使用你提供的 X-goog-api-key 请求头认证
    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": API_KEY
    }
    
    payload = {
        "contents": [{
            "parts": [{
                "text": prompt
            }]
        }]
    }
    
    for attempt in range(max_retries):
        try:
            time.sleep(1.0) 
            response = requests.post(API_URL, json=payload, headers=headers, timeout=15)
            response.raise_for_status()
            res_json = response.json()
            
            # 解析返回内容
            content = res_json['candidates'][0]['content']['parts'][0]['text']
            # 清理可能的 markdown 块包裹
            content = content.replace("```json", "").replace("```", "").strip()
            result = json.loads(content)
            
            return result.get('score', 0), result.get('insight', '无洞察')
            
        except Exception as e:
            print(f"[-] AI 解析报错 (尝试 {attempt+1}/{max_retries}): {e}")
            time.sleep(2)
            
    return 0, "解析超时或失败"

# ==========================================
# 3. 主流程
# ==========================================
def main():
    if not API_KEY:
        print("致命错误：找不到 LLM_API_KEY 环境变量！")
        return

    final_data = {
        "update_time": datetime.now().isoformat(), 
        "total_scanned": 0,
        "articles": []
    }
    
    for category, urls in SOURCES.items():
        print(f"\n>>> 开始抓取分类: {category}")
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
                            "publish_time": getattr(entry, 'published', datetime.now().isoformat())
                        })
            except Exception as e:
                print(f"[!] 抓取 {url} 失败: {e}")
                continue

    final_data["articles"] = sorted(final_data["articles"], key=lambda x: x["score"], reverse=True)
    
    with open("feed.json", "w", encoding="utf-8") as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n=== 执行完毕！筛选出 {len(final_data['articles'])} 条干货。 ===")

if __name__ == "__main__":
    main()
