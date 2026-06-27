---
name: offline-rag-rss-dashboard
description: Build offline-cached RAG systems on CPU, parse live XML RSS feeds, and serve them via a sleek glassmorphic Flask web interface.
---

# Offline RAG RSS Dashboard

This skill governs the construction of offline Retrieval-Augmented Generation (RAG) search engines, native parsing of XML RSS news feeds, and rendering them through interactive, glassmorphic Flask web servers.

## Workflow Blueprints

### 1. Offline Model Caching
Keep both sentence embedding models (e.g. `all-MiniLM-L6-v2`) and generation LLMs (e.g. `Qwen2.5-3B-Instruct` in bfloat16 mode) in-memory inside the Flask service to minimize latency on local CPU systems.

### 2. Native XML RSS Parsing
Avoid third-party parsing dependencies. Implement direct python RSS XML retrieval:
```python
import urllib.request
import xml.etree.ElementTree as ET

def fetch_rss_feed(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        xml_data = response.read()
    
    root = ET.fromstring(xml_data)
    articles = []
    for item in root.findall('.//item'):
        title = item.find('title').text
        link = item.find('link').text
        articles.append({"title": title, "link": link})
    return articles
```

### 3. Glassmorphic CSS Aesthetics
Serve responsive dashboard layouts featuring modern dark slates, `backdrop-filter: blur(12px)` for frosted glass, shimmering skeletons, and smooth micro-hover transitions to maximize presentation value.
