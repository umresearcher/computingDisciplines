/* The story screens, title to score (S01-S14). Each screen is a function that returns HTML. Exposes TS.screens. */
(function (TS) {
  "use strict";
  var E = window.FeedEngine;
  var STORIES = window.STORIES;
  var DAY1 = TS.game.DAY1, DAY2 = TS.game.DAY2, MIA_DAYS = TS.game.MIA_DAYS, DAY3 = TS.game.DAY3, state = TS.game.state, baseline = TS.game.baseline, sumR = TS.game.sumR, feed1 = TS.game.feed1, feed2 = TS.game.feed2, feed3 = TS.game.feed3, miaHistory = TS.game.miaHistory;
  var fmtDate = TS.util.fmtDate, signed = TS.util.signed, esc = TS.util.esc, count = TS.util.count, NUM = TS.util.NUM;
  var avatar = TS.art.avatar, figure = TS.art.figure, head = TS.art.head, PEOPLE = TS.art.PEOPLE;
  var LOGO = TS.art.ICONS.LOGO, ARROW = TS.art.ICONS.ARROW;
  var hud = TS.ui.hud, bubbles = TS.ui.bubbles, nextBtn = TS.ui.nextBtn, choiceCard = TS.ui.choiceCard, charCol = TS.ui.charCol, headRow = TS.ui.headRow, stage = TS.ui.stage, debater = TS.ui.debater, relBadge = TS.ui.relBadge, storyMeta = TS.ui.storyMeta, feedCard = TS.ui.feedCard, facts = TS.ui.facts, whatIf = TS.ui.whatIf, morganOnClicks = TS.ui.morganOnClicks;

  var SCREENS = {};

  SCREENS.title = function () {
    var stars = '<svg class="sky" viewBox="0 0 1280 760" preserveAspectRatio="xMidYMin slice" aria-hidden="true"><g class="stars-a" fill="#F6F2E7"><circle cx="96" cy="130" r="1.6"/><circle cx="182" cy="232" r="1.2"/><circle cx="250" cy="104" r="2"/><circle cx="318" cy="330" r="1.3"/><circle cx="404" cy="176" r="1.1"/><circle cx="140" cy="420" r="1.4"/><circle cx="870" cy="96" r="1.3"/><circle cx="960" cy="300" r="1.8"/><circle cx="1010" cy="420" r="1.2"/><circle cx="1164" cy="290" r="1.5"/><circle cx="1220" cy="420" r="1.1"/><circle cx="1090" cy="530" r="1.3"/><circle cx="70" cy="600" r="1.2"/><circle cx="230" cy="690" r="1.3"/><circle cx="1180" cy="640" r="1.4"/></g><g class="stars-b" fill="#DCD8FF"><circle cx="60" cy="280" r="1.2"/><circle cx="226" cy="380" r="1.6"/><circle cx="356" cy="64" r="1.2"/><circle cx="300" cy="210" r="1"/><circle cx="920" cy="190" r="1.6"/><circle cx="1100" cy="200" r="1.1"/><circle cx="1210" cy="90" r="1.4"/><circle cx="1000" cy="560" r="1.2"/><circle cx="110" cy="520" r="1.1"/><circle cx="1240" cy="540" r="1.2"/></g><path d="M190 170 l3 7 7 3 -7 3 -3 7 -3 -7 -7 -3 7 -3 Z" fill="#FFD23F"/><path d="M1070 360 l2.5 6 6 2.5 -6 2.5 -2.5 6 -2.5 -6 -6 -2.5 6 -2.5 Z" fill="#FFD23F"/><path d="M150 660 l2.5 6 6 2.5 -6 2.5 -2.5 6 -2.5 -6 -6 -2.5 6 -2.5 Z" fill="#FFD23F"/></svg>';
    var clouds = '<div class="cloud" style="left:5%;top:720px;width:240px;height:48px;background:rgba(185,178,255,.15)"></div><div class="cloud" style="left:10%;top:690px;width:124px;height:48px;background:rgba(185,178,255,.15)"></div><div class="cloud" style="right:5%;top:880px;width:270px;height:52px;background:rgba(246,242,231,.12)"></div><div class="cloud" style="right:11%;top:848px;width:134px;height:52px;background:rgba(246,242,231,.12)"></div>';
    return '<div class="screen title-page">' + stars + '<div class="moon" aria-hidden="true"></div><div class="shoot" aria-hidden="true"></div><div class="shoot two" aria-hidden="true"></div>' + clouds +
      '<div class="top"><div class="brand">' + LOGO + '<b>TownSquare</b></div><span>A game about news feeds · 5–8 min</span></div>' +
      '<main class="hero" tabindex="-1"><span class="eyebrow">Spring 2020 · a town stuck at home</span><h1>You Own the <em>Feed</em></h1>' +
      "<p>You run the news app everyone in town reads. Make three decisions about how it ranks real 2020 headlines, see what each reader ends up with, and find out what kind of feed you built.</p>" +
      '<button class="btn" data-enter data-go="S01"><svg width="20" height="22" viewBox="0 0 22 24" aria-hidden="true"><path d="M3 2.5 L19.5 12 L3 21.5 Z" fill="currentColor" stroke="currentColor" stroke-width="3" stroke-linejoin="round"/></svg>Run <span class="key wide">Enter</span></button>' +
      '<span class="hint" style="font-size:15px">Three decisions · real headlines · single player</span></main>' +
      '<div class=\"city\"><img src=\"assets/city.svg\" width=\"594\" height=\"482\" alt=\"A small night-time town on a block of earth: a café, a campus clinic, city hall, apartments, houses, a park and a plaza with a big feed screen. Chat bubbles float above the buildings.\"></div>' +
      '<div class="title-foot">For political science students · Headlines from the ReCOVery COVID-19 news dataset (Jan–May 2020) · Readers, reactions and clicks are simulated</div></div>';
  };

  SCREENS.S01 = function () {
    return hud(1) + stage("", charCol("morgan", "pumped"),
      headRow("morgan", "pumped") + '<span class="eyebrow">Your first day</span>' +
      bubbles(["Hey! You must be the new hire. I'm Morgan — I run <b>TownSquare</b>, the news app everyone in town reads.",
        "It's spring 2020. Everyone's stuck at home, and dozens of COVID-19 stories land on our app every day.",
        "Your job: write the rules that decide <b>which three stories each reader sees first.</b>",
        "Two goals: keep people reading, <b>and</b> give them a feed that's actually useful."]) +
      '<div class="spacer"></div>' + nextBtn("S02", "How does it work?", "Press Enter to continue"));
  };

  SCREENS.S02 = function () {
    return hud(2) + stage("", charCol("jamie", "curious"),
      headRow("jamie", "curious") + '<span class="eyebrow">How to play</span><h1 class="question">Three decisions. Real headlines. Then your score.</h1>' +
      '<div class="steps-how">' +
      '<div class="step-card"><span class="num">1</span><b>Set a rule</b><p>You make three decisions about how the feed ranks stories. Each one adds a rule to your algorithm (the bar at the top).</p></div>' +
      '<div class="step-card"><span class="num">2</span><b>See a reader\'s feed</b><p>After each decision you see the 3 stories a reader would get, picked from real headlines of that week — and why each one made it.</p></div>' +
      '<div class="step-card"><span class="num">3</span><b>Get your score</b><p>Morgan watches clicks the whole time. At the end you also see how relevant, reliable and varied your feed was.</p></div></div>' +
      '<div class="realbox"><b class="h">What\'s real and what\'s simulated</b><ul>' +
      "<li><b>Real:</b> headlines, outlets, dates and source-reliability labels from the ReCOVery dataset (1,889 COVID-19 news articles, Jan–May 2020).</li>" +
      "<li><b>Simulated:</b> the readers, their predicted reactions (estimated from headline wording) and clicks.</li>" +
      "<li>Topics are tagged automatically from each headline, so a few may look off.</li></ul></div>" +
      '<div class="spacer"></div>' + nextBtn("S03", "Meet the readers", ""));
  };

  SCREENS.S03 = function () {
    var cards = [["mia", "Mia", "Medical student", ["health", "schools"]], ["leo", "Leo", "Runs a café", ["business", "lockdown rules"]], ["sam", "Sam", "Local journalist", ["politics", "Chicago news"]]].map(function (r) {
      return '<div class="reader-card">' + avatar(r[0], 64) + "<div><b>" + r[1] + '</b><div class="role">' + r[2] + '</div></div><div class="hint" style="font-size:14px">Follows</div><div class="chips">' + r[3].map(function (c) { return '<span class="chip-outline">' + c + "</span>"; }).join("") + "</div></div>";
    }).join("");
    return hud(3) + stage("", charCol("jamie", "curious"),
      headRow("jamie", "curious") + '<span class="eyebrow">Meet the readers</span>' +
      bubbles(["These are three of our readers. Each one follows different topics.", "Every day dozens of stories come in, but a feed shows only the <b>top 3</b>. Your rules decide which 3."]) +
      '<div class="reader-cards">' + cards + '</div><div class="spacer"></div>' + nextBtn("S04", "Make decision 1", ""));
  };

  SCREENS.S04 = function () {
    state._deciding = "1";
    var html = hud(4) + stage("", charCol("morgan", "impatient"),
      headRow("morgan", "impatient", "Head of Platform") + '<span class="eyebrow">Decision 1 of 3</span>' +
      '<h1 class="question">What should decide which stories reach the top 3?</h1>' +
      '<div class="bubble">We launch Monday. I\'m not saying pick reactions… I\'m just saying reactions pay for the servers.</div>' +
      '<div class="spacer"></div><div class="choices">' +
      choiceCard(1, "Predicted reactions", "Stories most likely to get likes, comments and shares go first.", "Example: a dramatic headline beats a dry update.", "", "rule1:reactions") +
      choiceCard(2, "The reader's interests", "Stories on topics the reader follows go first.", "Example: Leo follows business, so business news goes first.", "", "rule1:interests") +
      '</div><div class="hint">No wrong answer — you\'ll see the result right away. Press 1 or 2.</div>');
    state._deciding = null;
    return html;
  };

  function resultScreen(scene, n, reader, day, list, other, otherLabel, who, mood, line, extra, next, nextLabel) {
    var f = facts(reader, list, day);
    var pool = E.pool(STORIES, day).length;
    return hud(scene, f.clicks) + stage("wide-char", charCol(who, mood, { bubble: line, plate: false }),
      headRow(who, mood, PEOPLE[who].role, line) +
      '<span class="eyebrow">Result of decision ' + n + "</span>" +
      '<h1 class="scene-title">' + E.READERS[reader].name + "'s feed on " + fmtDate(day) + "</h1>" +
      feedCard(reader, day, list, pool) +
      '<div class="takeaway"><span class="h">What happened</span><ul>' + f.html + "</ul>" + (extra ? "<p>" + extra + "</p>" : "") + "</div>" +
      whatIf(reader, other, otherLabel) + '<div class="spacer"></div>' + nextBtn(next, nextLabel, ""));
  }

  SCREENS.S05 = function () {
    var list = feed1(), f = facts("leo", list, DAY1);
    var other = feed1(state.rule1 === "reactions" ? "interests" : "reactions");
    var extra = state.rule1 === "reactions"
      ? (f.m < 3 ? "Ranking by reactions put attention-grabbing stories on top, even ones Leo doesn't follow." : "Ranking by reactions put the loudest stories on top.")
      : "Ranking by interests gave Leo his topics. Notice that this rule doesn't look at who published a story.";
    return resultScreen(5, 1, "leo", DAY1, list, other, state.rule1 === "reactions" ? "The reader's interests" : "Predicted reactions",
      "morgan", f.clicks >= 0 ? "pumped" : "impatient", morganOnClicks(f.clicks), extra, "S06", "Next: a word from the newsroom");
  };

  function loudestLow(day) {
    var low = E.pool(STORIES, day).filter(function (s) { return s.rel === 0; });
    low.sort(function (a, b) { return b.r - a.r; });
    return low[0];
  }

  SCREENS.S06 = function () {
    var s = loudestLow(DAY1);
    var inFeed = feed1().some(function (c) { return c.s === s; });
    return hud(6) + stage("", charCol("alex", "dry"),
      headRow("alex", "dry") + '<span class="eyebrow">Before decision 2</span>' +
      '<div class="bubble">Alex, newsroom editor. ' + (inFeed ? "This story is in Leo's top 3 right now." : "This is the most-reacted story from a low-reliability outlet this week.") + " Take a look.</div>" +
      '<div class="story-card"><span class="title">' + esc(s.t) + '</span><span class="meta">' + storyMeta(s, "leo", false) + "</span></div>" +
      '<div class="bubble">The dataset rates <b>' + esc(s.p) + "</b> as a low-reliability outlet — its track record is poor. That doesn't prove this story is false. It means nobody should trust it before checking.</div>" +
      '<div class="legend-row"><span>' + relBadge({ rel: 1 }) + "outlet with a good track record</span><span>" + relBadge({ rel: 0 }) + "outlet with a poor track record</span></div>" +
      '<p class="datanote" style="margin:0">Reliability is a label for the whole outlet in the ReCOVery dataset, not a fact-check of each article.</p>' +
      '<div class="spacer"></div>' + nextBtn("S07", "Make decision 2", ""));
  };

  SCREENS.S07 = function () {
    state._deciding = "2";
    var html = hud(7) + '<main class="stage single" tabindex="-1"><div class="content" style="padding-top:32px">' +
      '<span class="eyebrow">Decision 2 of 3</span><h1 class="question">Should the feed favor reliable outlets?</h1>' +
      '<div class="debate">' +
      debater("morgan", "impatient", "Your rule is working. Change it, and who knows what happens to clicks?") +
      debater("alex", "dry", "Give reliable outlets an edge. Nobody's deleting anything — the rest just ranks lower.") +
      '</div><div class="spacer" style="min-height:20px"></div><div class="choices">' +
      choiceCard(1, "Favor reliable outlets", "Stories from outlets with a good track record rank higher. Low-reliability outlets rank lower.", "Nothing is removed. A low-reliability story can still appear if nothing better fits.", "Effect on clicks: unknown", "rule2:boost") +
      choiceCard(2, "Keep ranking as it is", "Who published a story doesn't count. Your rule 1 stays in charge.", "The feed keeps doing what it already does.", "Effect on clicks: predictable", "rule2:keep") +
      '</div><div class="hint">Both are real trade-offs. Press 1 or 2.</div></div></main>';
    state._deciding = null;
    return html;
  };

  SCREENS.S08 = function () {
    var list = feed2(), f = facts("leo", list, DAY2);
    var other = feed2(state.rule2 === "boost" ? "keep" : "boost");
    var otherLow = count(other, function (c) { return c.s.rel === 0; });
    var extra = state.rule2 === "boost"
      ? (otherLow > 0 ? "Without this rule, " + NUM[otherLow] + " of Leo's top 3 would have come from low-reliability outlets." : "This week the rule changed little: Leo's top stories were already from reliable outlets.")
      : (f.low > 0 ? "With reliability ignored, " + NUM[f.low] + " of Leo's top 3 come from low-reliability outlets." : "This week Leo's top stories happened to come from reliable outlets anyway.");
    var who = state.rule2 === "boost" ? "morgan" : "alex";
    var line = who === "morgan" ? morganOnClicks(f.clicks) : "Popular and checked aren't the same thing. Noted.";
    return resultScreen(8, 2, "leo", DAY2, list, other, state.rule2 === "boost" ? "Keep ranking as it is" : "Favor reliable outlets",
      who, who === "morgan" ? (f.clicks >= 40 ? "pumped" : "impatient") : "dry", line, extra, "S09", "Next: something odd in Mia's feed");
  };

  SCREENS.S09 = function () {
    var mh = miaHistory();
    var rows = mh.days.map(function (d, i) {
      return '<span class="day">Day ' + (i + 1) + "</span>" + d.feed.map(function (c) { return '<span class="tile ' + (c.s.topic === mh.days[2].feed[0].s.topic ? "same" : "other") + '" title="' + esc(c.s.t) + '">' + c.s.topic + "</span>"; }).join("");
    }).join("");
    var main = mh.days[2].feed[0].s.topic;
    var lastSame = count(mh.days[2].feed, function (c) { return c.s.topic === main; });
    return hud(9) + stage("", charCol("jamie", "check"),
      headRow("jamie", "check") + '<span class="eyebrow">Before decision 3</span>' +
      '<div class="bubble">Something odd in Mia\'s feed. Each day she opens her top story, and the feed learns from that. Let\'s check what happened.</div>' +
      '<figure class="topic-chart" style="margin:0"><div class="top"><figcaption style="font-weight:700;font-size:18px">Topics in Mia\'s top 3, ' + fmtDate(MIA_DAYS[0], { month: "short" }) + "–" + fmtDate(MIA_DAYS[2], { month: "short" }) + '</figcaption><span class="legend"><i></i>' + main + '</span></div><div class="grid3">' + rows + "</div></figure>" +
      '<div class="bubble">By day 3, ' + (lastSame === 3 ? "<b>all three</b>" : "<b>" + lastSame + " of 3</b>") + " of Mia's stories are about " + main + ". She keeps opening " + main + ", so the feed keeps serving " + main + ".</div>" +
      '<div class="spacer"></div>' + nextBtn("S10", "Make decision 3", ""));
  };

  SCREENS.S10 = function () {
    state._deciding = "3";
    var html = hud(10) + '<main class="stage single" tabindex="-1"><div class="content" style="padding-top:32px">' +
      '<span class="eyebrow">Decision 3 of 3</span><h1 class="question">Should the feed mix in topics a reader didn\'t ask for?</h1>' +
      '<div class="debate">' +
      debater("morgan", "pumped", "She opens it, she gets more of it. That's the whole deal. Why fight it?") +
      debater("jamie", "curious", "Or mix in a few other topics — and see what she does with them.") +
      '</div><div class="spacer" style="min-height:20px"></div><div class="choices">' +
      choiceCard(1, "Add variety", "Limit repeats: the top 3 should cover more than one topic.", "Mia may see politics or business next to health.", "Fewer sure clicks", "rule3:variety") +
      choiceCard(2, "Keep the focus", "Show more of what each reader opened before.", "Mia keeps getting the topic she clicks most.", "Proven, predictable clicks", "rule3:focus") +
      '</div><div class="hint">Last decision. Press 1 or 2.</div></div></main>';
    state._deciding = null;
    return html;
  };

  SCREENS.S11 = function () {
    var list = feed3(), f = facts("mia", list, DAY3);
    var other = feed3(state.rule3 === "variety" ? "focus" : "variety");
    var extra = state.rule3 === "variety"
      ? "Mia now sees " + f.topics.length + " different topics instead of one. Some may be less clicky — that's the trade-off."
      : "Mia gets more of what she already opens. Anything outside " + f.topics.join(" and ") + " never reaches her top 3.";
    var who = state.rule3 === "variety" ? "jamie" : "morgan";
    var line = who === "jamie" ? "Fewer sure clicks, maybe. But look how much more of the news she sees!" : morganOnClicks(f.clicks);
    return resultScreen(11, 3, "mia", DAY3, list, other, state.rule3 === "variety" ? "Keep the focus" : "Add variety",
      who, who === "jamie" ? "curious" : (f.clicks >= 0 ? "pumped" : "impatient"), line, extra, "S12", "Two quick questions, then your score");
  };

  function checkScreen(n, scene, q, opts, correct, key, who, mood, fb, next, nextLabel) {
    var answered = state[key], chosen = state[key + "Pick"];
    var html = hud(scene) + '<main class="check-wrap" tabindex="-1" style="padding-inline:16px"><div style="display:flex;align-items:center;gap:14px">' + head(who, mood, 64) +
      '<div style="display:flex;flex-direction:column;gap:2px"><span class="eyebrow">Quick check · ' + n + ' of 2</span><span style="font-size:17px">Your answer counts toward the “understanding” part of your score.</span></div></div>' +
      '<h1 class="check-q">' + q + '</h1><div class="answers">' +
      opts.map(function (o, i) {
        var k = i + 1, cls = "answer", verdict = "";
        if (answered) {
          if (k === correct) { cls += " right"; verdict = chosen === k ? "Your answer · correct" : "Correct answer"; }
          else if (k === chosen) { cls += " wrong"; verdict = "Your answer"; }
          else cls += " dim";
        }
        return '<button class="' + cls + '" ' + (answered ? "disabled" : 'data-key="' + k + '" data-act="' + key + ":" + k + '"') + '><span class="key">' + k + "</span>" + o + (verdict ? '<span class="verdict">' + verdict + "</span>" : "") + "</button>";
      }).join("") + "</div>";
    if (answered) html += '<div class="feedback">' + head(who, mood, 72) + '<div class="bubble"><b>' + PEOPLE[who].name + ":</b> " + (answered === "right" ? fb.right : fb.wrong) + "</div></div>" + nextBtn(next, nextLabel, "");
    else html += '<div class="hint">Pick one — you\'ll see the answer right away. Press 1 or 2.</div>';
    return html + "</main>";
  }
  SCREENS.S12 = function () {
    return checkScreen(1, 12, "Why did Mia's top 3 end up on one topic?", ["She asked to see only that topic.", "The feed kept rewarding what she had opened before."], 2, "check1", "jamie", "curious",
      { right: "Right. The feed learned from her clicks and kept serving more of the same.", wrong: "Not quite — she never asked. The feed kept rewarding what she had opened before." }, "S13", "Next question");
  };
  SCREENS.S13 = function () {
    return checkScreen(2, 13, "A story comes from an outlet the dataset rates as low reliability. What does that tell you?", ["The story is false.", "The outlet has a poor track record, so check the claim before trusting or sharing it."], 2, "check2", "alex", "look",
      { right: "Exactly. A low-reliability label is a warning about the outlet, not proof about one story. Check it elsewhere first.", wrong: "Careful — the label is about the outlet's track record, not proof that this story is false. Check it elsewhere first." }, "S14", "See your score");
  };

  function score() {
    var feeds = [feed1(), feed2(), feed3()];
    var met = E.metrics(feeds);
    var right = (state.check1 === "right" ? 1 : 0) + (state.check2 === "right" ? 1 : 0);
    var slots = 9;
    var rel = Math.round(met.relevance * 25), relia = Math.round(met.reliability * 25), vari = Math.round(met.variety * 25), und = Math.round(right * 12.5);
    var matchN = Math.round(met.relevance * slots), relN = Math.round(met.reliability * slots);
    var b = sumR(baseline(DAY1)) + sumR(baseline(DAY2)) + sumR(baseline(DAY3));
    var clicks = Math.round((met.reactions - b) / b * 100);
    return {
      total: rel + relia + vari + und, clicks: clicks,
      parts: [
        { name: "Relevance", n: rel, why: matchN + " of the 9 stories you showed matched what the reader follows." },
        { name: "Reliability", n: relia, why: relN + " of the 9 stories came from outlets rated reliable." },
        { name: "Variety", n: vari, why: state.rule3 === "variety" ? "You added variety, so feeds covered more topics." : "Feeds often repeated one topic. Mia's last top 3 was " + facts("mia", feeds[2], DAY3).topics.join(", ") + "." },
        { name: "Understanding", n: und, why: right === 2 ? "Both quick checks right." : right === 1 ? "One quick check right." : "No quick checks right — the Rule lab can help." }
      ]
    };
  }

  SCREENS.S14 = function () {
    var s = score();
    var morganLine = s.clicks >= 30 ? "Clicks up " + s.clicks + "%! I'm framing this." : s.clicks >= 0 ? "Up " + s.clicks + "%. I'll take it. Barely." : "Clicks down " + Math.abs(s.clicks) + "%. We need to talk.";
    return hud(14, s.clicks) + '<main class="result-wrap" tabindex="-1" style="padding-inline:64px">' +
      '<div><span class="eyebrow">Your score</span><h1 class="result-title">Here\'s the feed you built.</h1><p class="hint" style="font-size:19px;margin:6px 0 0">Morgan watched clicks. Your score shows the other half of the job: was the feed useful?</p></div>' +
      '<div class="result-grid"><section class="watched" aria-label="Clicks"><span class="eyebrow" style="font-size:14px">What Morgan watched</span><span class="big">' + signed(s.clicks) + '</span><b style="font-size:18px">Clicks vs. newest-first</b><span class="hint">Total predicted reactions on your 9 stories, compared with simply showing the newest stories. Simulated. Not part of your score.</span><div class="spacer"></div><div style="display:flex;align-items:center;gap:12px">' + head("morgan", "pumped", 64) + '<span style="font-size:17px;line-height:1.4">Morgan: “' + morganLine + '”</span></div></section>' +
      '<section class="scorecard" aria-label="Feed score"><div style="display:flex;align-items:baseline;gap:14px;flex-wrap:wrap"><span class="eyebrow" style="font-size:14px;color:var(--muted-ink)">Feed score</span><span class="total">' + s.total + "<small> / 100</small></span></div>" +
      s.parts.map(function (p) { return '<div class="part"><b>' + p.name + '</b><div class="bar"><i style="width:' + (p.n / 25 * 100) + '%"></i></div><span class="n">' + p.n + "</span><span></span><p>" + p.why + "</p></div>"; }).join("") +
      '<details class="how"><summary>How this is scored</summary><ul><li>Each part is out of 25, from the 9 stories you showed (Leo on Mar 30, Leo on May 6, Mia on May 9).</li><li>Relevance: share of stories that match the reader. Reliability: share from reliable outlets. Variety: how many different topics each top 3 covered. Understanding: the two quick checks.</li><li>Clicks never count toward the score.</li></ul></details></section></div>' +
      '<div class="next-row"><button class="btn" data-enter data-go="LAB">Try your own rules in the lab ' + ARROW + '</button><button class="btn-ghost" data-act="restart">Play again</button></div></main>';
  };

  TS.screens = SCREENS;
})(window.TS = window.TS || {});
