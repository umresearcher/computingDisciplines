/* Game state, the three decisions as engine weights, and the feeds each decision produces. Exposes TS.game and TS.util. */
(function (TS) {
  "use strict";
  var E = window.FeedEngine;
  var STORIES = window.STORIES;

  var DAY1 = "2020-03-30", DAY2 = "2020-05-06", MIA_DAYS = ["2020-05-06", "2020-05-07", "2020-05-08"], DAY3 = "2020-05-09";
  var TOTAL = 14;

  var FRESH = { screen: "title", rule1: null, rule2: null, rule3: null, check1: null, check2: null, check1Pick: null, check2Pick: null, lab: null };
  var cache = {};                          // ranked feeds, keyed by rule + day + reader
  var state = Object.assign({}, FRESH);   // one object for the whole game; reset() refills it in place
  function reset(saved) {
    Object.keys(state).forEach(function (k) { delete state[k]; });
    Object.assign(state, FRESH, saved || {});
    Object.keys(cache).forEach(function (k) { delete cache[k]; });
  }

  function weights(r1, r2, r3) {
    return {
      wR: r1 === "interests" ? 1 : 4,
      wI: r1 === "interests" ? 4 : 1,
      wE: r2 === "boost" ? 3 : 0,
      wV: r3 === "variety" ? 3 : 0,
      wH: r3 === "focus" ? 3 : r3 === "variety" ? 0 : 2
    };
  }
  function feedFor(key, reader, day, w, history) {
    var k = key + JSON.stringify(w) + (history || []).join(",");
    if (!cache[k]) cache[k] = E.rank(E.pool(STORIES, day), reader, w, history);
    return cache[k];
  }
  function baseline(day) { return E.newestFirst(E.pool(STORIES, day)).slice(0, 3); }
  function sumR(list) { return list.reduce(function (a, x) { return a + (x.s ? x.s.r : x.r); }, 0); }
  function clicksVs(feed, day) { var b = sumR(baseline(day)); return Math.round((sumR(feed) - b) / b * 100); }

  function feed1(r1) { return feedFor("1", "leo", DAY1, weights(r1 || state.rule1)); }
  function feed2(r2) { return feedFor("2", "leo", DAY2, weights(state.rule1, r2 || state.rule2)); }
  function miaHistory() {
    var hist = [], days = [];
    MIA_DAYS.forEach(function (d) {
      var f = feedFor("m" + d, "mia", d, weights(state.rule1, state.rule2, null), hist.slice());
      days.push({ day: d, feed: f });
      hist.push(f[0].s.topic);
    });
    return { hist: hist, days: days };
  }
  function feed3(r3) { var h = miaHistory().hist; return feedFor("3", "mia", DAY3, weights(state.rule1, state.rule2, r3 || state.rule3), h); }

  function fmtDate(iso, opts) {
    return new Date(iso + "T12:00:00Z").toLocaleDateString("en-US", Object.assign({ timeZone: "UTC", month: "long", day: "numeric" }, opts || {}));
  }
  function signed(n) { return (n > 0 ? "+" : n < 0 ? "−" : "") + Math.abs(n) + "%"; }
  function esc(s) { return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;"); }
  function count(list, fn) { return list.filter(fn).length; }
  var NUM = ["no", "one", "two", "all three"];
  function lastClicks() {
    if (state.rule3) return clicksVs(feed3(), DAY3);
    if (state.rule2) return clicksVs(feed2(), DAY2);
    if (state.rule1) return clicksVs(feed1(), DAY1);
    return null;
  }

  TS.game = { DAY1: DAY1, DAY2: DAY2, MIA_DAYS: MIA_DAYS, DAY3: DAY3, TOTAL: TOTAL, FRESH: FRESH, state: state, reset: reset, weights: weights,
    feedFor: feedFor, baseline: baseline, sumR: sumR, clicksVs: clicksVs, feed1: feed1, feed2: feed2, feed3: feed3, miaHistory: miaHistory, lastClicks: lastClicks };
  TS.util = { fmtDate: fmtDate, signed: signed, esc: esc, count: count, NUM: NUM };
})(window.TS = window.TS || {});
