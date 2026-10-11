"""Turn the ReCOVery CSV into the small story table the game uses.

Per story: id, title, publisher, date, topic, reliable (1/0/None), bias, reactions (simulated, in thousands).
Usage: python3 -I prep_data.py news.csv stories.json
"""
import csv
import json
import re
import sys
from datetime import datetime

# Topic = first rule whose keywords appear in the title (then the first 400 chars of the body).
TOPICS = [
    ("schools", r"\b(school|schools|student|students|campus|college|colleges|universit\w*|class(es|room)?|teacher\w*|graduat\w*|tuition|exam\w*)\b"),
    ("business", r"\b(econom\w*|business\w*|stocks?|stock market|markets?|jobs?|unemploy\w*|layoff\w*|retail\w*|restaurant\w*|bar|bars|shop\w*|store\w*|compan\w*|stimulus|loan\w*|recession|wall street|dow|bank\w*|airline\w*|industr\w*|small business\w*|workers?|employ\w*|rent|sales|amazon|tesla|uber)\b"),
    ("health", r"\b(vaccin\w*|hospital\w*|doctor\w*|nurse\w*|patient\w*|symptom\w*|treat\w*|drug\w*|cure\w*|test(s|ing|ed)?|mask\w*|ventilator\w*|icu|immun\w*|antibod\w*|hydroxychloroquine|chloroquine|remdesivir|malaria|medic\w*|health\w*|infect\w*|clinic\w*|pharmac\w*|nursing homes?|cases|deaths?|died|dies|toll|cdc|fda|who|scientists?)\b"),
    ("lockdown rules", r"\b(lockdown\w*|stay[- ]at[- ]home|shelter[- ]in[- ]place|quarantin\w*|reopen\w*|restriction\w*|social distanc\w*|curfew|ban\w*|closure\w*|closed|close|travel)\b"),
    ("politics", r"\b(trump|biden|president\w*|governor\w*|pritzker|lightfoot|mayor|congress\w*|senat\w*|white house|democrat\w*|republican\w*|election\w*|vote\w*|primar\w*|government\w*|minister\w*|pompeo|pelosi|cuomo|kremlin|putin|policy|official\w*|city council|aldermen|alderman)\b"),
]
# readers' interests (topic names above); "local" = Chicago stories, a tag on top of the topic
READERS = {"mia": ["health", "schools"], "leo": ["business", "lockdown rules"], "sam": ["politics", "local"]}


def topic_of(title, body):
    """Most keyword hits wins: headline hits count 3x, first 600 chars of the article 1x."""
    t, b = title.lower(), (body or "")[:600].lower()
    best, best_score = "world", 0
    for name, rx in TOPICS:
        sc = 3 * len(re.findall(rx, t)) + len(re.findall(rx, b))
        if sc > best_score:
            best, best_score = name, sc
    return best


# Simulated "predicted reactions": a transparent headline-only model, the same for every outlet.
EMOTION = r"\b(shock\w*|horrif\w*|terrif\w*|panic\w*|chaos|crisis|deadly|killer|kill\w*|slam\w*|blast\w*|destroy\w*|outrage\w*|furious|fear\w*|warn\w*|exposed?|secret\w*|truth|lie|lies|hoax|cover[- ]?up|bombshell|explod\w*|surge\w*|soar\w*|plung\w*|crash\w*|collapse\w*|disaster\w*|nightmare|miracle|cure|breaking|urgent|must|never|worst|best|first|only|bioweapon|lab|bill gates|5g|conspira\w*)\b"
POLITICAL = r"\b(trump|biden|china|chinese|wuhan|pelosi|cuomo|putin|democrats?|republicans?|fauci|who)\b"


def reactions(title):
    t = title.lower()
    score = 1.0
    score += 1.6 * len(re.findall(EMOTION, t))
    score += 0.9 * len(re.findall(POLITICAL, t))
    score += 0.8 if "?" in title else 0
    score += 1.0 if "!" in title else 0
    score += 0.6 if re.search(r"\byou(r)?\b", t) else 0
    score += 0.5 if re.search(r"\d", t) else 0
    score += 0.7 if len(title) < 60 else 0
    caps = len(re.findall(r"\b[A-Z]{3,}\b", title))
    score += 0.6 * min(caps, 2)
    # stable per-story wobble so ties break the same way every time
    h = sum(ord(c) for c in title) % 97 / 97
    return round(0.6 + score * score * 0.9 + h * 1.5, 1)   # thousands of reactions


def main(src, out):
    rows = list(csv.DictReader(open(src, encoding="utf-8")))
    stories = []
    for r in rows:
        title = " ".join((r["title"] or "").split())
        if not title:
            continue
        d = datetime.strptime(r["publish_date"], "%m/%d/%y").date()
        rel = r["reliability"].strip()
        stories.append({
            "id": int(r["news_id"]),
            "t": title,
            "p": r["publisher"].strip(),
            "d": d.isoformat(),
            "topic": topic_of(title, r["body_text"]),
            "local": 1 if r["publisher"].strip() == "Chicago Sun-Times" or "chicago" in title.lower() or "illinois" in title.lower() else 0,
            "rel": None if rel == "" else int(float(rel)),
            "bias": r["political_bias"].strip() or None,
            "country": r["country"].strip() or None,
            "r": reactions(title),
        })
    stories.sort(key=lambda s: (s["d"], s["id"]))
    json.dump(stories, open(out, "w"), ensure_ascii=True, separators=(",", ":"))
    print(len(stories), "stories")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
