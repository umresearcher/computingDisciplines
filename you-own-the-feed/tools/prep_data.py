"""Turn the ReCOVery CSV into the story table the game uses, and rate every story on the game's scale.

Per story: id, t = title, p = publisher, d = date, topic, local (Chicago), rel = outlet reliability (1/0/None),
r = simulated predicted reactions (thousands), and three levels on the 1-3 scale (1 = Low, 2 = Mid, 3 = High):
  R  reactions:   which third of all headlines its predicted reactions fall in
  L  reliability: 3 = reliable outlet rated center or center-leaning, 2 = reliable outlet rated strongly
                  left/right (or not rated), 1 = low-reliability outlet (ReCOVery reliability + political-bias labels)
  I  interest:    [Mia, Leo, Sam] - 3 = a topic the reader follows, 2 = a related topic, 1 = anything else
The game picks the stories whose levels are closest to the dial settings.
Usage: python3 -I prep_data.py news.csv ../js/data/stories.js
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
RELATED = {"mia": ["lockdown rules"], "leo": ["politics"], "sam": ["lockdown rules", "schools"]}
READER_ORDER = ["mia", "leo", "sam"]


def interest(story, reader):
    if story["topic"] in READERS[reader] or ("local" in READERS[reader] and story["local"]):
        return 3
    return 2 if story["topic"] in RELATED[reader] else 1


def rate(stories):
    """Add the R / L / I levels. Reactions: cut points at the 1/3 and 2/3 quantiles of the whole dataset."""
    rs = sorted(s["r"] for s in stories)
    lo, hi = rs[len(rs) // 3], rs[2 * len(rs) // 3]
    for s in stories:
        s["R"] = 1 if s["r"] < lo else (2 if s["r"] < hi else 3)
        if s["rel"] == 0:
            s["L"] = 1                                  # low-reliability outlet
        elif s["rel"] == 1 and s["bias"] not in ("Left", "Right"):
            s["L"] = 3                                  # reliable outlet, rated center / center-leaning
        else:
            s["L"] = 2                                  # reliable but rated strongly one-sided, or not rated
        s["I"] = [interest(s, rd) for rd in READER_ORDER]
    return lo, hi


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
    lo, hi = rate(stories)
    keep = ("id", "t", "p", "d", "topic", "local", "rel", "r", "R", "L", "I")
    slim = [{k: s[k] for k in keep} for s in stories]
    body = json.dumps(slim, ensure_ascii=True, separators=(",", ":"))
    if out.endswith(".js"):   # a script, so the game also runs from a local file
        body = ("/* " + str(len(slim)) + " headlines from the ReCOVery COVID-19 news dataset (Jan-May 2020), made by tools/prep_data.py.\n"
                "   R / L / I = reactions / reliability / interest [Mia, Leo, Sam] on the 1-3 scale (Low, Mid, High). */\n"
                "window.STORIES = " + body + ";\n")
    open(out, "w").write(body)
    print(len(stories), "stories; reactions cut points", lo, hi)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
