/* Rule lab (bonus round after the score): four dials re-rank a real week of headlines live. Adds TS.screens.LAB, exposes TS.lab. */
(function (TS) {
  "use strict";
  var E = window.FeedEngine;
  var STORIES = window.STORIES;
  var state = TS.game.state, weights = TS.game.weights;
  var fmtDate = TS.util.fmtDate, signed = TS.util.signed, esc = TS.util.esc, NUM = TS.util.NUM;
  var avatar = TS.art.avatar, head = TS.art.head;
  var LOGO = TS.art.ICONS.LOGO;
  var hud = TS.ui.hud, stage = TS.ui.stage, WHY = TS.ui.WHY, storyMeta = TS.ui.storyMeta, facts = TS.ui.facts;

  var SCREENS = TS.screens;
  var LAB_DIALS = [
    { key: "wR", label: "Reactions", hint: "Boost stories likely to get likes and shares" },
    { key: "wI", label: "Interests", hint: "Boost topics the reader follows" },
    { key: "wE", label: "Reliability", hint: "Lift reliable outlets, sink low-reliability ones" },
    { key: "wV", label: "Variety", hint: "Avoid repeating one topic in the top 3" }
  ];
  var WORDS = ["Off", "Low", "Mid", "High", "Max"];
  var LAB_DAYS = [["2020-03-30", "Week of Mar 30"], ["2020-04-20", "Week of Apr 20"], ["2020-05-06", "Week of May 6"]];
  function labStart() {
    var w = weights(state.rule1, state.rule2, state.rule3);
    return { wR: Math.min(4, w.wR), wI: Math.min(4, w.wI), wE: w.wE, wV: w.wV, wH: 0 };
  }
  function labRank(L, w) { return E.rankLab(E.pool(STORIES, L.day), L.reader, w || L.w, 5); }
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
      return '<div class="post v2' + (i >= 3 ? " missed" : "") + (fresh ? " fresh" : "") + '"><span class="rank">' + (i + 1) + '</span><div><span class="title">' + esc(c.s.t) + '</span><span class="meta">' + storyMeta(c.s, L.reader, true) + '</span><span class="why">Why it\'s here: <b>' + WHY[c.why](c, L.reader) + "</b></span>" + (chip ? '<span style="display:block;margin-top:6px">' + chip + "</span>" : "") + "</div></div>" +
        (i === 2 ? '<div class="missed-label">Just missed the top 3</div>' : "");
    }).join("");
    return '<section class="feed" style="max-width:none" aria-label="' + name + '\'s feed"><div class="feed-head"><div class="who">' + avatar(L.reader, 36) + "<b>" + name + "'s top 3</b><time>" + fmtDate(L.day, { weekday: "short", year: "numeric" }) + "</time></div>" +
      '<span class="sub">Picked from ' + poolSize + " real headlines published " + fmtDate(E.addDays(L.day, -6), { month: "short" }) + "\u2013" + fmtDate(L.day, { month: "short" }) + ", 2020 \u00B7 " + name + " follows " + E.READERS[L.reader].likeLabel + "</span></div>" + rows + "</section>";
  }
  SCREENS.LAB = function () {
    if (!state.lab) state.lab = { reader: "leo", day: "2020-05-06", w: labStart(), prev: null, change: null };
    var L = state.lab, poolSize = E.pool(STORIES, L.day).length;
    var all = labRank(L), list = all.slice(0, 3);
    var f = facts(L.reader, list, L.day), name = E.READERS[L.reader].name;
    var status;
    if (L.prev && L.change) {
      var ids = list.map(function (c) { return c.s.id; });
      var fresh = ids.filter(function (id) { return L.prev.indexOf(id) < 0; }).length;
      var reordered = !fresh && ids.join() !== L.prev.join();
      status = "<b>" + L.change + ":</b> " + (fresh ? fresh + " new " + (fresh === 1 ? "story" : "stories") + " in " + name + "\u2019s top 3." : reordered ? "same three stories, in a new order." : "no change in the top 3. This dial doesn\u2019t matter much for these stories this week \u2014 try a bigger jump, another dial or another week.");
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
    else insight = "Let\u2019s check: push one dial to Max and watch what moves.";
    var dials = LAB_DIALS.map(function (d) {
      return '<div class="dial" role="group" aria-label="' + d.label + '"><div class="top"><b>' + d.label + '</b><span class="word">' + WORDS[L.w[d.key]] + '</span></div><span class="hint">' + d.hint + '</span><div class="steps">' +
        WORDS.map(function (word, i) { return '<button class="step" aria-pressed="' + (L.w[d.key] === i) + '" aria-label="' + d.label + ": " + word + '" data-act="dial:' + d.key + ":" + i + '">' + word + "</button>"; }).join("") + "</div></div>";
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
    return '<header class="hud"><div class="brand">' + LOGO + "<b>TownSquare</b><span>Rule lab</span></div><span class=\"hint\" style=\"font-size:15px\">Bonus round \u00B7 a sandbox, doesn't change your score</span><button class=\"btn-ghost\" style=\"min-height:44px;font-size:16px\" data-act=\"restart\">Play again</button></header>" +
      '<main class="lab-stage" tabindex="-1"><div style="display:flex;flex-direction:column;gap:10px;min-width:0"><div style="display:flex;align-items:flex-start;gap:12px;margin-bottom:4px">' + head("jamie", "curious", 56) +
      '<div class="bubble" style="font-size:18px;padding:10px 16px;border-radius:6px 16px 16px 16px;animation:none">' + insight + "</div></div>" + dials +
      '<div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-top:2px"><button class="btn-ghost" style="min-height:44px;font-size:15px;border-color:var(--lilac)" data-act="labreset">Back to your rules</button><button class="btn-ghost" style="min-height:44px;font-size:15px;border-color:var(--lilac)" data-act="laboff">All dials off</button></div>' +
      '<span class="hint" style="font-size:13px">\u201CBack to your rules\u201D uses your three decisions. With all dials off, the feed simply shows the newest stories.</span></div>' +
      '<div style="display:flex;flex-direction:column;gap:14px;min-width:0"><div class="tabs">' + tabs + '</div><div class="lab-status" role="status" aria-live="polite">' + status + "</div>" + labFeed(L, all, poolSize) +
      '<div class="meters">' + meters + '</div><span class="datanote">Headlines and reliability labels: ReCOVery dataset. Predicted reactions and clicks are simulated from headline wording. In the lab, every dial uses the same 0\u20134 scale.</span></div></main>';
  };

  TS.lab = { labRank: labRank, labStart: labStart, LAB_DIALS: LAB_DIALS, WORDS: WORDS, LAB_DAYS: LAB_DAYS };
})(window.TS = window.TS || {});
