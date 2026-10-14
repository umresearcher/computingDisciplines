/* Rule lab (bonus round after the score). Every story is rated Low / Mid / High on reactions, interest and reliability;
   the four dials say which level you want, and the feed shows the stories that fit best. Adds TS.screens.LAB, exposes TS.lab. */
(function (TS) {
  "use strict";
  var E = window.FeedEngine;
  var STORIES = window.STORIES;
  var state = TS.game.state, target = TS.game.target;
  var fmtDate = TS.util.fmtDate, signed = TS.util.signed, esc = TS.util.esc, NUM = TS.util.NUM;
  var avatar = TS.art.avatar, head = TS.art.head;
  var LOGO = TS.art.ICONS.LOGO;
  var hud = TS.ui.hud, stage = TS.ui.stage, WHY = TS.ui.WHY, storyMeta = TS.ui.storyMeta, facts = TS.ui.facts;

  var SCREENS = TS.screens;
  // levels: what Low / Mid / High mean for each dial ({name} = the reader)
  var LAB_DIALS = [
    { key: "reactions", label: "Reactions", chip: "Reactions", levels: ["calmest third of headlines", "middle third", "loudest third of headlines"] },
    { key: "interests", label: "Interests", chip: "Interest", levels: ["topics {name} doesn\u2019t follow", "topics close to {name}\u2019s", "topics {name} follows"] },
    { key: "reliability", label: "Reliability", chip: "Reliability", levels: ["low-reliability outlets", "reliable, but strongly one-sided", "reliable, center-leaning outlets"] },
    { key: "topics", label: "Variety", chip: "Topics", levels: ["top 3 on one topic", "top 3 on two topics", "top 3 on three topics"] }
  ];
  var WORDS = ["Low", "Mid", "High"];   // dial values are 1, 2, 3
  var LAB_DAYS = [["2020-03-30", "Week of Mar 30"], ["2020-04-20", "Week of Apr 20"], ["2020-05-06", "Week of May 6"]];
  // your three decisions as dial positions; anything you didn't decide starts at Mid
  function labStart() {
    var t = target(state.rule1, state.rule2, state.rule3), w = {};
    LAB_DIALS.forEach(function (d) { w[d.key] = t[d.key] || 2; });
    return w;
  }
  // L.last = the dial moved last: it must match, the others are matched as closely as the week allows
  function labRank(L, w) { return E.select(E.pool(STORIES, L.day), L.reader, w || L.w, { n: 5, focus: L.last }); }
  function meaning(d, v, reader) { return d.levels[v - 1].replace(/\{name\}/g, E.READERS[reader].name); }
  // one chip per characteristic: the story's level, ticked when it is the level you asked for
  function levelChips(c, L) {
    return LAB_DIALS.slice(0, 3).map(function (d) {
      var have = c.lv[d.key], want = L.w[d.key], ok = have === want;
      return '<span class="lvl ' + (ok ? "ok" : "off") + '" title="You asked for ' + WORDS[want - 1] + '">' + (ok ? "\u2713 " : "") + d.chip + " " + WORDS[have - 1] + "</span>";
    }).join("");
  }
  function fitLine(c) {
    if (!c.miss.length) return "fits all three settings";
    var names = c.miss.map(function (k) { return LAB_DIALS.filter(function (d) { return d.key === k; })[0].chip.toLowerCase(); });
    return "closest fit \u2014 " + names.join(" and ") + " not as asked";
  }
  function labFeed(L, all, poolSize) {
    var name = E.READERS[L.reader].name, prev = L.prev;
    var rows = all.map(function (c, i) {
      var chip = "";
      if (prev && i < 3) {
        var was = prev.indexOf(c.s.id);
        if (was < 0) chip = '<span class="move new">New in top 3</span>';
        else if (was > i) chip = '<span class="move up">\u25B2 up ' + (was - i) + "</span>";
        else if (was < i) chip = '<span class="move down">\u25BC down ' + (i - was) + "</span>";
      }
      var fresh = prev && i < 3 && prev.indexOf(c.s.id) < 0;
      return '<div class="post v2' + (i >= 3 ? " missed" : "") + (fresh ? " fresh" : "") + '"><span class="rank">' + (i + 1) + '</span><div><span class="title">' + esc(c.s.t) + '</span><span class="meta">' + storyMeta(c.s, L.reader, true) + '</span><span class="lvls">' + levelChips(c, L) + '</span><span class="why">Why it\'s here: <b>' + fitLine(c) + "</b></span>" + (chip ? '<span style="display:block;margin-top:6px">' + chip + "</span>" : "") + "</div></div>" +
        (i === 2 ? '<div class="missed-label">Just missed the top 3</div>' : "");
    }).join("");
    return '<section class="feed" style="max-width:none" aria-label="' + name + '\'s feed"><div class="feed-head"><div class="who">' + avatar(L.reader, 36) + "<b>" + name + "'s top 3</b><time>" + fmtDate(L.day, { weekday: "short", year: "numeric" }) + "</time></div>" +
      '<span class="sub">Picked from ' + poolSize + " real headlines published " + fmtDate(E.addDays(L.day, -6), { month: "short" }) + "\u2013" + fmtDate(L.day, { month: "short" }) + ", 2020 \u00B7 " + name + " follows " + E.READERS[L.reader].likeLabel + "</span></div>" + rows + "</section>";
  }
  SCREENS.LAB = function () {
    if (!state.lab) state.lab = { reader: "leo", day: "2020-05-06", w: labStart(), prev: null, change: null };
    if (state.lab.w.wR !== undefined) state.lab.w = labStart();   // saved from the old 0-4 dials
    var L = state.lab, poolSize = E.pool(STORIES, L.day).length;
    var all = labRank(L), list = all.slice(0, 3);
    var f = facts(L.reader, list, L.day), name = E.READERS[L.reader].name;
    var status;
    if (L.prev && L.change) {
      var ids = list.map(function (c) { return c.s.id; });
      var fresh = ids.filter(function (id) { return L.prev.indexOf(id) < 0; }).length;
      var reordered = !fresh && ids.join() !== L.prev.join();
      var already = "";
      if (!fresh && !reordered && L.last) {
        var dl = LAB_DIALS.filter(function (d) { return d.key === L.last; })[0], want = L.w[L.last];
        var fits = L.last === "topics" ? f.topics.length === want : list.every(function (c) { return c.lv[L.last] === want; });
        if (fits) already = L.last === "topics" ? "the top 3 already covered " + want + (want === 1 ? " topic" : " topics") + ", so nothing had to move." : "all three stories already had " + dl.chip.toLowerCase() + " " + WORDS[want - 1] + ", so nothing had to move.";
      }
      status = "<b>" + L.change + ":</b> " + (fresh ? fresh + " new " + (fresh === 1 ? "story" : "stories") + " in " + name + "\u2019s top 3." : reordered ? "same three stories, in a new order." : already || (L.change === "Back to your rules" ? "the dials were already set to your rules." : "no change \u2014 this week has no stories that fit better. Try another dial or another week."));
    } else if (L.change) {
      status = "<b>" + L.change + "</b> \u2014 a new set of stories to rank.";
    } else {
      status = "Push a dial and watch the feed. Stories that change are marked.";
    }
    var insight;
    if (f.low > 0) insight = NUM[f.low].charAt(0).toUpperCase() + NUM[f.low].slice(1) + " of " + name + "\u2019s top 3 " + (f.low === 1 ? "comes" : "come") + " from a low-reliability outlet. Try turning up Reliability.";
    else if (f.topics.length === 1) insight = "All three stories are about " + f.topics[0] + ". What is " + name + " missing? Try Variety.";
    else if (f.m === 0) insight = "None of these match what " + name + " follows. Try Interests.";
    else if (f.clicks < 0) insight = "Relevant and reliable \u2014 but clicks are below newest-first. That\u2019s the trade-off.";
    else insight = "Every story has a Low, Mid or High level for each dial. Change one and watch which stories fit.";
    var dials = LAB_DIALS.map(function (d) {
      var v = L.w[d.key];
      return '<div class="dial' + (L.last === d.key ? " last" : "") + '" role="group" aria-label="' + d.label + '"><div class="top"><b>' + d.label + '</b><span class="word">' + (L.last === d.key ? '<span class="must">must match</span>' : "") + WORDS[v - 1] + '</span></div><span class="hint">' + WORDS[v - 1] + " = " + meaning(d, v, L.reader) + '</span><div class="steps">' +
        WORDS.map(function (word, i) { return '<button class="step" aria-pressed="' + (v === i + 1) + '" aria-label="' + d.label + ": " + word + '" data-act="dial:' + d.key + ":" + (i + 1) + '">' + word + "</button>"; }).join("") + "</div></div>";
    }).join("");
    var tabs = '<span class="hint" style="font-size:15px">Whose feed?</span>' + ["mia", "leo", "sam"].map(function (id) {
      return '<button class="tab" aria-pressed="' + (L.reader === id) + '" data-act="reader:' + id + '"><i style="background:' + { mia: "#3DD6C4", leo: "#FF6B4A", sam: "#B9B2FF" }[id] + '"></i>' + E.READERS[id].name + "</button>";
    }).join("") + '<span class="hint" style="font-size:15px;margin-left:8px">Week:</span>' + LAB_DAYS.map(function (d) {
      return '<button class="tab" aria-pressed="' + (L.day === d[0]) + '" data-act="day:' + d[0] + '" style="padding:0 14px">' + d[1] + "</button>";
    }).join("");
    var met = E.metrics([list]);
    var meters = [
      ["Clicks vs. newest-first", signed(f.clicks), Math.max(0, Math.min(100, 50 + f.clicks / 2)), "var(--coral)"],
      ["Relevance", f.m + " / 3", f.m / 3 * 100, "var(--teal)"],
      ["Reliable outlets", Math.round(met.reliability * 3) + " / 3", met.reliability * 100, "var(--teal)"],
      ["Topics covered", f.topics.length + " / 3", f.topics.length / 3 * 100, "var(--teal)"]
    ].map(function (m) { return '<div class="meter"><span>' + m[0] + "</span><b>" + m[1] + '</b><div class="bar"><i style="width:' + m[2] + "%;background:" + m[3] + '"></i></div></div>'; }).join("");
    return '<header class="hud"><button class="brand" data-act="restart" title="Back to the start" aria-label="TownSquare: back to the start">' + LOGO + "<b>TownSquare</b><span>Rule lab</span></button><span class=\"hint\" style=\"font-size:15px\">Bonus round \u00B7 a sandbox, doesn't change your score</span><button class=\"btn-ghost\" style=\"min-height:44px;font-size:16px\" data-act=\"restart\">Play again</button></header>" +
      '<main class="lab-stage" tabindex="-1"><div style="display:flex;flex-direction:column;gap:10px;min-width:0"><div style="display:flex;align-items:flex-start;gap:12px;margin-bottom:4px">' + head("jamie", "curious", 56) +
      '<div class="bubble" style="font-size:18px;padding:10px 16px;border-radius:6px 16px 16px 16px;animation:none">' + insight + "</div></div>" + dials +
      '<div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-top:2px"><button class="btn-ghost" style="min-height:44px;font-size:15px;border-color:var(--lilac)" data-act="labreset">Back to your rules</button></div>' +
      '<span class="hint" style="font-size:13px">\u201CBack to your rules\u201D sets the dials from your three decisions (anything you didn\u2019t decide starts at Mid).</span></div>' +
      '<div style="display:flex;flex-direction:column;gap:14px;min-width:0"><div class="tabs">' + tabs + '</div><div class="lab-status" role="status" aria-live="polite">' + status + "</div>" + labFeed(L, all, poolSize) +
      '<div class="meters">' + meters + '</div><span class="datanote">Headlines, reliability and political-bias labels: ReCOVery dataset. Predicted reactions and clicks are simulated from headline wording. Levels: reactions = which third of all headlines; reliability = the outlet\u2019s reliability and bias labels; interest = the reader\u2019s topics.</span></div></main>';
  };

  TS.lab = { labRank: labRank, labStart: labStart, LAB_DIALS: LAB_DIALS, WORDS: WORDS, LAB_DAYS: LAB_DAYS };
})(window.TS = window.TS || {});
