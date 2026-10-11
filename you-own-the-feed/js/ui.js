/* Small HTML builders shared by the screens: HUD, algorithm bar, bubbles, buttons, feed cards. Exposes TS.ui. */
(function (TS) {
  "use strict";
  var E = window.FeedEngine;
  var TOTAL = TS.game.TOTAL, state = TS.game.state, clicksVs = TS.game.clicksVs, lastClicks = TS.game.lastClicks;
  var fmtDate = TS.util.fmtDate, signed = TS.util.signed, esc = TS.util.esc, count = TS.util.count;
  var avatar = TS.art.avatar, figure = TS.art.figure, head = TS.art.head, PEOPLE = TS.art.PEOPLE;
  var LOGO = TS.art.ICONS.LOGO, ARROW = TS.art.ICONS.ARROW;

  function hud(scene, clicks) {
    var c;
    if (clicks === undefined) clicks = lastClicks();
    if (clicks === null) c = '<div class="clicks" title="Clicks: predicted reactions on the stories you show, compared with showing the newest stories first."><span>Clicks</span><strong>—</strong><small>simulated</small></div>';
    else c = '<div class="clicks ' + (clicks >= 0 ? "up" : "down") + '" title="Predicted reactions on the stories you show, compared with showing the newest stories first."><span>Clicks</span><strong>' + (clicks >= 0 ? "▲ " : "▼ ") + signed(clicks) + "</strong><small>vs. newest-first · simulated</small></div>";
    return '<header class="hud"><button class="brand" data-act="restart" title="Back to the start" aria-label="TownSquare: back to the start">' + LOGO + "<b>TownSquare</b><span>Algorithm desk</span></button>" +
      '<div class="progress"><span><span class="label-long">Scene </span>' + scene + " of " + TOTAL + "</span>" +
      '<div class="track" role="progressbar" aria-label="Progress" aria-valuemin="0" aria-valuemax="' + TOTAL + '" aria-valuenow="' + scene + '"><div class="fill" style="width:' + (scene / TOTAL * 100).toFixed(1) + '%"></div></div></div>' + c + "</header>" + algoBar();
  }
  function algoBar(deciding) {
    deciding = deciding || state._deciding;
    var slots = [
      ["1", "Ranks by", state.rule1 && (state.rule1 === "reactions" ? "predicted reactions" : "reader interests")],
      ["2", "Sources", state.rule2 && (state.rule2 === "boost" ? "favor reliable outlets" : "reliability ignored")],
      ["3", "Variety", state.rule3 && (state.rule3 === "variety" ? "mix in other topics" : "more of the same")]
    ];
    return '<nav class="algo" aria-label="Your algorithm so far"><span class="lbl">Your algorithm</span>' + slots.map(function (s) {
      var cls = s[2] ? "done" : (deciding === s[0] ? "now" : "");
      var val = s[2] ? "<b>" + s[2] + "</b>" : (deciding === s[0] ? "deciding now" : "not set");
      return '<span class="slot ' + cls + '"><span class="n">' + s[0] + "</span>" + s[1] + ": " + val + "</span>";
    }).join("") + "</nav>";
  }
  function bubbles(lines) { return lines.map(function (l) { return '<div class="bubble">' + l + "</div>"; }).join(""); }
  function nextBtn(go, label, hintText) {
    return '<div class="next-row"><button class="btn" data-enter data-go="' + go + '">' + label + " " + ARROW + "</button>" + (hintText ? '<span class="hint">' + hintText + "</span>" : "") + "</div>";
  }
  function choiceCard(key, title, desc, example, tradeoff, action) {
    return '<button class="choice big" data-key="' + key + '" data-act="' + action + '"><span class="t"><span class="key">' + key + "</span>" + title + '</span><span class="d">' + desc + "</span>" +
      (example ? '<span class="ex">' + example + "</span>" : "") + (tradeoff ? '<span class="tradeoff">' + tradeoff + "</span>" : "") + "</button>";
  }
  function charCol(who, mood, opts) {
    opts = opts || {};
    var p = PEOPLE[who];
    return '<div class="char-col">' + (opts.bubble ? '<div class="bubble float-bubble">' + opts.bubble + "</div>" : "") +
      '<div class="disc" style="background:' + p.disc + '"></div>' + figure(who, mood) +
      (opts.plate === false ? "" : '<div class="plate"><b>' + p.name + "</b><span>" + p.role + "</span></div>") + "</div>";
  }
  function headRow(who, mood, sub, line) {
    var p = PEOPLE[who];
    return '<div class="head-row">' + head(who, mood, 56) + "<div><b>" + p.name + "</b><span>" + (sub || p.role) + "</span></div></div>" + (line ? '<div class="bubble head-row-bubble">' + line + "</div>" : "");
  }
  function stage(cls, left, right) { return '<main class="stage ' + (cls || "") + '" tabindex="-1">' + left + '<div class="content">' + right + "</div></main>"; }
  function debater(who, mood, line) {
    return '<div class="debater">' + head(who, mood, 120) + '<div class="col"><span class="name">' + PEOPLE[who].name + '</span><div class="bubble">' + line + "</div></div></div>";
  }

  function relBadge(s) {
    if (s.rel === 1) return '<span class="badge rel">Reliable outlet</span>';
    if (s.rel === 0) return '<span class="badge low">Low-reliability outlet</span>';
    return '<span class="badge unk">Reliability not rated</span>';
  }
  var WHY = {
    reactions: function (c) { return "high predicted reactions (" + c.s.r + "K)"; },
    interests: function (c, reader) { return "matches what " + E.READERS[reader].name + " follows"; },
    reliability: function () { return "reliable outlet, boosted by your rule"; },
    habit: function (c, reader) { return "more of what " + E.READERS[reader].name + " opened before"; },
    recency: function () { return "one of the newest stories (no dial pushed it)"; }
  };
  function storyMeta(s, reader, showMatch) {
    return '<span class="outlet">' + esc(s.p) + "</span><span>" + fmtDate(s.d, { month: "short" }) + "</span>" + relBadge(s) + '<span class="badge topic">' + s.topic + "</span>" +
      (showMatch && E.matches(s, reader) ? '<span class="badge match">' + E.READERS[reader].name + " follows this</span>" : "") +
      "<span>" + s.r + "K predicted reactions</span>";
  }
  function feedCard(reader, day, list, poolSize) {
    var name = E.READERS[reader].name;
    var from = E.addDays(day, -6);
    return '<section class="feed" aria-label="' + name + '\'s top 3"><div class="feed-head"><div class="who">' + avatar(reader, 36) + "<b>" + name + "'s top 3</b><time>" + fmtDate(day, { weekday: "short", year: "numeric" }) + "</time></div>" +
      '<span class="sub">Picked from ' + poolSize + " real headlines published " + fmtDate(from, { month: "short" }) + "–" + fmtDate(day, { month: "short" }) + ", 2020 · " + name + " follows " + E.READERS[reader].likeLabel + "</span></div>" +
      list.map(function (c, i) {
        return '<div class="post v2"><span class="rank">' + (i + 1) + '</span><div><span class="title">' + esc(c.s.t) + '</span><span class="meta">' + storyMeta(c.s, reader, true) + '</span><span class="why">Why it\'s here: <b>' + WHY[c.why](c, reader) + "</b></span></div></div>";
      }).join("") + "</section>";
  }
  function facts(reader, list, day) {
    var name = E.READERS[reader].name;
    var m = count(list, function (c) { return c.match; });
    var low = count(list, function (c) { return c.s.rel === 0; });
    var topics = []; list.forEach(function (c) { if (topics.indexOf(c.s.topic) < 0) topics.push(c.s.topic); });
    return { m: m, low: low, topics: topics, clicks: clicksVs(list, day),
      html: "<li><b>" + m + " of 3</b> match what " + name + " follows.</li><li><b>" + low + " of 3</b> come from low-reliability outlets.</li><li><b>" + topics.length + (topics.length === 1 ? " topic" : " topics") + "</b> in the top 3: " + topics.join(", ") + ".</li>" };
  }
  function whatIf(reader, list, label) {
    return '<details class="whatif"><summary>What if you had picked “' + label + '”?</summary><ol>' + list.map(function (c) {
      return "<li>" + esc(c.s.t) + "<br><span>" + esc(c.s.p) + " · " + (c.s.rel === 1 ? "reliable outlet" : c.s.rel === 0 ? "low-reliability outlet" : "not rated") + " · " + c.s.topic + " · " + c.s.r + "K predicted reactions</span></li>";
    }).join("") + "</ol></details>";
  }
  function morganOnClicks(clicks) {
    if (clicks >= 40) return "Clicks up " + clicks + "% compared with newest-first! People <i>love</i> this feed.";
    if (clicks >= 0) return "Clicks up " + clicks + "%. Fine. I was hoping for more.";
    return "Clicks down " + Math.abs(clicks) + "%. I'm writing that down.";
  }

  TS.ui = { hud: hud, algoBar: algoBar, bubbles: bubbles, nextBtn: nextBtn, choiceCard: choiceCard, charCol: charCol, headRow: headRow, stage: stage, debater: debater, relBadge: relBadge, WHY: WHY, storyMeta: storyMeta, feedCard: feedCard, facts: facts, whatIf: whatIf, morganOnClicks: morganOnClicks };
})(window.TS = window.TS || {});
