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

  // the reader follows this story's topic (interest level High)
  function matches(story, reader) { return story.I[IDX[reader]] === 3; }

  // Newest first: what the feed would show with no ranking rule at all.
  function newestFirst(list) {
    return list.slice().sort(function (a, b) { return a.d < b.d ? 1 : a.d > b.d ? -1 : b.id - a.id; });
  }

  /* Every story carries three levels on the same 1-3 scale (1 = Low, 2 = Mid, 3 = High), set in tools/prep_data.py:
       R = predicted reactions, L = reliability, I = interest for [Mia, Leo, Sam].
     A feed setting ("target") says which level you want for each one, plus how many topics the top 3 should cover:
       { reactions: 1-3 | null, interests: 1-3 | null, reliability: 1-3 | null, topics: 1-3 | null }   (null = not set)
     select() picks the stories whose levels are closest to the target:
       1. distance = sum over the set levels of |story level - wanted level|   (0 = exact match);
          the dial moved last (opts.focus) is a must-have (counts 10x), so the feed always answers the change you just made
       2. ties: the topic the reader opened more often before (history), then more predicted reactions, then newer
       3. topics: the top 3 must cover exactly that many topics (1 = one topic, 3 = three different topics) */
  var IDX = { mia: 0, leo: 1, sam: 2 };
  var DIMS = ["reactions", "interests", "reliability"];
  var LEVELS = ["Low", "Mid", "High"];

  function levelsOf(s, reader) { return { reactions: s.R, interests: s.I[IDX[reader]], reliability: s.L }; }

  function select(list, reader, target, opts) {
    opts = opts || {};
    var n = opts.n || 3, history = opts.history || [], focus = opts.focus;
    var perTopic = {};
    list.forEach(function (s) { perTopic[s.topic] = (perTopic[s.topic] || 0) + 1; });
    var cands = list.map(function (s) {
      var lv = levelsOf(s, reader), dist = 0, miss = [];
      DIMS.forEach(function (k) {
        if (target[k] == null) return;
        var d = Math.abs(lv[k] - target[k]);
        dist += k === focus ? d * 10 : d;
        if (d) miss.push(k);
      });
      var habit = history.filter(function (t) { return t === s.topic; }).length;
      return { s: s, lv: lv, dist: dist, miss: miss, habit: habit, match: lv.interests === 3 ? 1 : 0 };
    });
    cands.sort(function (a, b) {
      return a.dist - b.dist || b.habit - a.habit || b.s.r - a.s.r || (a.s.d < b.s.d ? 1 : a.s.d > b.s.d ? -1 : b.s.id - a.s.id);
    });

    var picked = [], topics = [], k = target.topics;
    for (var i = 0; i < cands.length && picked.length < 3; i++) {
      var c = cands[i];
      if (k) {
        var isNew = topics.indexOf(c.s.topic) < 0;
        var after = topics.length + (isNew ? 1 : 0), slotsLeft = 3 - picked.length - 1;
        if (after > k || after + slotsLeft < k) continue;           // would make the topic count impossible
        if (k === 1 && perTopic[c.s.topic] < 3) continue;            // one topic needs at least 3 stories
        if (isNew) topics.push(c.s.topic);
      }
      picked.push(c);
    }
    // "just missed": the next best stories, then anything left if the topic rule could not be met
    cands.forEach(function (c) { if (picked.length < n && picked.indexOf(c) < 0) picked.push(c); });

    picked.forEach(function (c) {
      c.why = "recency";
      for (var j = 0; j < DIMS.length; j++) {
        if (target[DIMS[j]] != null && c.lv[DIMS[j]] === target[DIMS[j]] && target[DIMS[j]] === 3) { c.why = DIMS[j]; break; }
      }
      if (c.why === "recency" && c.habit > 0) c.why = "habit";
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

  var api = { READERS: READERS, LEVELS: LEVELS, DIMS: DIMS, addDays: addDays, pool: pool, levelsOf: levelsOf, select: select, newestFirst: newestFirst, matches: matches, metrics: metrics };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.FeedEngine = api;
})(this);
