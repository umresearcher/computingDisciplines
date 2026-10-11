// Feed engine for You Own the Feed. Pure functions; works in the browser (window.FeedEngine) and Node.
(function (root) {
  var READERS = {
    mia: { name: "Mia", likes: ["health", "schools"], likeLabel: "health, schools" },
    leo: { name: "Leo", likes: ["business", "lockdown rules"], likeLabel: "business, lockdown rules" },
    sam: { name: "Sam", likes: ["politics", "local"], likeLabel: "politics, Chicago news" }
  };

  function addDays(iso, n) {
    var d = new Date(iso + "T12:00:00Z");
    d.setUTCDate(d.getUTCDate() + n);
    return d.toISOString().slice(0, 10);
  }

  // Stories published in the 7 days up to and including `day`.
  function pool(stories, day, days) {
    var from = addDays(day, -((days || 7) - 1));
    return stories.filter(function (s) { return s.d >= from && s.d <= day; });
  }

  function matches(story, reader) {
    var likes = READERS[reader].likes;
    return likes.indexOf(story.topic) >= 0 || (likes.indexOf("local") >= 0 && story.local === 1);
  }

  // Newest first: what the feed would show with no ranking rule at all.
  function newestFirst(list) {
    return list.slice().sort(function (a, b) { return a.d < b.d ? 1 : a.d > b.d ? -1 : b.id - a.id; });
  }

  /* Rank one day's pool for one reader.
     w.wR reactions, w.wI interests, w.wE reliability, w.wV variety, w.wH "more of what you clicked".
     history = topics this reader opened on earlier days (most recent last). */
  function rank(list, reader, w, history) {
    history = history || [];
    var scored = list.map(function (s) {
      var m = matches(s, reader) ? 1 : 0;
      var rel = s.rel === 1 ? 1 : s.rel === 0 ? -1 : 0;
      var h = history.filter(function (t) { return t === s.topic; }).length;
      var parts = {
        reactions: w.wR * Math.min(1, s.r / 20),
        interests: w.wI * m,
        reliability: w.wE * rel * 0.6,
        habit: w.wH * Math.min(h, 3) * 0.45
      };
      return { s: s, match: m, parts: parts, base: parts.reactions + parts.interests + parts.reliability + parts.habit };
    });
    var picked = [];
    var lastTopics = history.slice(-3);
    for (var k = 0; k < 3; k++) {
      var best = null, bestV = -Infinity;
      scored.forEach(function (c) {
        if (picked.indexOf(c) >= 0) return;
        var same = picked.filter(function (p) { return p.s.topic === c.s.topic; }).length;
        var stale = lastTopics.indexOf(c.s.topic) >= 0 ? 1 : 0;
        var v = c.base - w.wV * (0.45 * same + 0.25 * stale);
        if (v > bestV || (v === bestV && c.s.r > best.s.r)) { bestV = v; best = c; }
      });
      if (!best) break;
      best.final = bestV;
      picked.push(best);
    }
    // the one rule that did the most to put each story there
    picked.forEach(function (c) {
      var p = c.parts, top = "reactions", val = p.reactions;
      ["interests", "reliability", "habit"].forEach(function (k) { if (p[k] > val) { val = p[k]; top = k; } });
      c.why = top;
    });
    var base = newestFirst(list);
    picked.forEach(function (c) { c.was = base.indexOf(c.s) + 1; });
    return picked;
  }


  /* Rule lab ranking: every dial is on the same 0..1 scale so each one visibly matters.
     reactions = percentile of predicted reactions within this week's pool; interests = 1 if the reader follows the topic;
     reliability = +1 reliable outlet, -1 low-reliability outlet; a small recency term means "all dials off" = newest first. */
  function rankLab(list, reader, w, n) {
    var byR = list.slice().sort(function (a, b) { return a.r - b.r; });
    var newest = newestFirst(list);
    var N = Math.max(1, list.length - 1);
    var scored = list.map(function (s) {
      var pct = byR.indexOf(s) / N, rec = 1 - newest.indexOf(s) / N;
      var m = matches(s, reader) ? 1 : 0, rel = s.rel === 1 ? 1 : s.rel === 0 ? -1 : 0;
      var parts = { reactions: w.wR * pct, interests: w.wI * m, reliability: w.wE * rel * 0.8, recency: 0.6 * rec };
      return { s: s, match: m, parts: parts, base: parts.reactions + parts.interests + parts.reliability + parts.recency };
    });
    var picked = [];
    for (var k = 0; k < (n || 5); k++) {
      var best = null, bestV = -Infinity;
      scored.forEach(function (c) {
        if (picked.indexOf(c) >= 0) return;
        var same = picked.filter(function (p) { return p.s.topic === c.s.topic; }).length;
        var v = c.base - w.wV * 0.6 * same;
        if (v > bestV) { bestV = v; best = c; }
      });
      if (!best) break;
      picked.push(best);
    }
    picked.forEach(function (c) {
      var p = c.parts, top = "recency", val = p.recency;
      ["reactions", "interests", "reliability"].forEach(function (k) { if (p[k] > val) { val = p[k]; top = k; } });
      c.why = top;
    });
    return picked;
  }

  function metrics(feeds) {
    // feeds: array of picked lists (each 3 stories). Returns 0..1 shares.
    var slots = 0, rel = 0, match = 0, reactions = 0, topicsPerFeed = 0;
    feeds.forEach(function (f) {
      var topics = [];
      f.forEach(function (c) {
        slots++;
        if (c.s.rel === 1) rel++;
        if (c.match) match++;
        reactions += c.s.r;
        if (topics.indexOf(c.s.topic) < 0) topics.push(c.s.topic);
      });
      topicsPerFeed += (topics.length - 1) / 2;
    });
    return { relevance: match / slots, reliability: rel / slots, variety: topicsPerFeed / feeds.length, reactions: reactions };
  }

  var api = { READERS: READERS, addDays: addDays, pool: pool, rank: rank, rankLab: rankLab, newestFirst: newestFirst, matches: matches, metrics: metrics };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.FeedEngine = api;
})(this);
