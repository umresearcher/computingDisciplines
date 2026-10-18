/* Game rules. No drawing here: this file only changes numbers.
   One turn = news arrives -> you pick answer 0 or 1 -> votes move -> the rival gains a little. */
(function (CG) {
  "use strict";

  var START = { D: 45, R: 45, I: 44 };      // starting votes, %
  var DRIFT = { D: 1.1, R: 1.1, I: 0.85 };  // the rival gains this much every turn
  var WIN = 52, LOSE = 35;
  var DELAY = 2;                             // follow-ups land this many turns later

  var PARTY = {
    D: { name: "Democrat", color: "#4F8BFF", dark: "#1E3A78", light: "#9DBBFF" },
    R: { name: "Republican", color: "#F2545B", dark: "#5C1A20", light: "#FFB0B5" },
    I: { name: "Independent", color: "#A88BFF", dark: "#3B2A6E", light: "#CBB8FF" }
  };

  function round1(x) { return Math.round(x * 10) / 10; }

  /* The rest of the field, for the "Your race" tab. The game only tracks YOUR share;
     everyone else is filled in around it: two small third-party candidates, a shrinking
     group of undecided voters, and the main rival(s) get whatever is left. */
  var OTHERS = {
    D: { name: "Elaine Porter", party: "Democrat", color: "#4F8BFF", dark: "#1E3A78", look: { skin: "#F6D3B3", hair: "bun", hairColor: "#8B3A1E", glasses: true } },
    R: { name: "Richard Hale", party: "Republican", color: "#F2545B", dark: "#5C1A20", look: { skin: "#E8B48F", hair: "side", hairColor: "#B8B8BE", glasses: false } },
    L: { name: "Wes Calder", party: "Libertarian", color: "#E5B94E", dark: "#4A3A12", look: { skin: "#C68B5E", hair: "short", hairColor: "#1A1423", glasses: false } },
    G: { name: "Mara Lind", party: "Green", color: "#5FB57A", dark: "#1C3A26", look: { skin: "#8D5A3B", hair: "curly", hairColor: "#1A1423", glasses: false } }
  };

  // everyone's share at one point of the race (step = index in run.history)
  function fieldAt(party, mine, step) {
    var undecided = round1(Math.max(2, 7 - step * 0.35));
    var lib = round1(2.4 + 0.4 * Math.sin(step * 1.7));
    var green = round1(1.5 + 0.3 * Math.sin(step * 2.3 + 1));
    var rest = 100 - mine - lib - green - undecided;
    var out = { me: mine, L: lib, G: green, undecided: undecided };
    if (party === "I") {               // an independent faces both major parties
      out.D = round1(rest * (0.51 + 0.03 * Math.sin(step * 1.3)));
      out.R = round1(rest - out.D);
    } else {
      out[party === "D" ? "R" : "D"] = round1(rest);
    }
    return out;
  }

  // the race now and one step ago, plus the history of the main rival (for the chart)
  function field(run) {
    var h = run.history, n = h.length - 1;
    var now = fieldAt(run.party, run.votes, n), before = fieldAt(run.party, h[Math.max(0, n - 1)], Math.max(0, n - 1));
    var rivalKey = run.party === "D" ? "R" : run.party === "R" ? "D" : (now.D >= now.R ? "D" : "R");
    var rival = h.map(function (v, i) { return fieldAt(run.party, v, i)[rivalKey]; });
    return { now: now, before: before, rivalKey: rivalKey, rivalHistory: rival };
  }

  function eventById(n) {
    for (var i = 0; i < window.EVENTS.length; i++) if (window.EVENTS[i].n === n) return window.EVENTS[i];
    return null;
  }

  function shuffle(list) {
    var a = list.slice();
    for (var i = a.length - 1; i > 0; i--) {
      var j = Math.floor(Math.random() * (i + 1));
      var t = a[i]; a[i] = a[j]; a[j] = t;
    }
    return a;
  }

  // every event this party can get, in random order (some events are only for D/R or only for I)
  function newDeck(party) {
    var ids = window.EVENTS.filter(function (e) { return e[party]; }).map(function (e) { return e.n; });
    return shuffle(ids);
  }

  function newRun(party) {
    var run = {
      party: party, votes: START[party], turn: 1, drift: DRIFT[party],
      deck: newDeck(party), current: null,
      pending: [],       // follow-ups waiting to land
      history: [START[party]],
      moments: [],       // every vote change, for the end screen
      cheers: 0, paidOff: 0, hurt: 0, replies: 0,
      last: null, later: null, over: null
    };
    drawNext(run);
    return run;
  }

  function drawNext(run) {
    if (!run.deck.length) run.deck = newDeck(run.party);
    run.current = run.deck.pop();
  }

  // Who comments and what they think. `k` is the answer (0 or 1), `now` the vote change.
  function reactions(party, n, k, now) {
    var T = window.TALK[n][k], lean = T[0];
    var people = { D: ["Dana", "Democratic voter"], R: ["Ray", "Republican voter"] };
    var likes = function (p) { return lean === "L" ? p === "D" : p === "R"; };
    // on party-coded answers a D or R base reacts to the lean; otherwise to the result
    var baseLikes = (party !== "I" && lean !== "N") ? likes(party) : now >= 0;
    var comments = [], why = [];

    var base = party === "I" ? { name: "Sky", tag: "your supporters", party: "I" } : { name: people[party][0], tag: "your base", party: party };
    comments.push({ name: base.name, tag: base.tag, party: base.party, text: baseLikes ? T[1] : T[2] });
    why.push({ who: party === "I" ? "Your supporters" : "Your base", good: baseLikes, text: baseLikes ? "liked " + T[3] + "." : "felt " + T[4] + "." });

    comments.push({ name: "Chris", tag: "undecided", party: null, text: T[5] });
    why.push({ who: "Undecided voters", good: null, text: T[6] + "." });

    if (lean !== "N") {
      var other = party === "I" ? (lean === "L" ? "R" : "D") : (party === "D" ? "R" : "D");
      var otherLikes = likes(other);
      comments.push({ name: people[other][0], tag: party === "I" ? people[other][1] : "rival supporter", party: other, text: otherLikes ? T[1] : T[2] });
      why.push({ who: other === "D" ? "Democratic voters" : "Republican voters", good: otherLikes, text: otherLikes ? "liked " + T[3] + "." : "felt " + T[4] + "." });
    }
    return { comments: comments, why: why };
  }

  function checkOver(run) {
    if (run.votes >= WIN) run.over = "win";
    else if (run.votes <= LOSE) run.over = "lose";
  }

  // The player picks answer k for the current news.
  function choose(run, k) {
    var e = eventById(run.current);
    var now = e[run.party][k];
    var said = k === 0 ? e.A : e.B;
    var follow = e.followup && e.followup[k];

    if (follow) run.pending.push({ due: run.turn + DELAY, outcomes: follow, n: e.n, said: said, turn: run.turn });

    run.votes = round1(run.votes + now - run.drift);
    run.history.push(run.votes);

    var r = reactions(run.party, e.n, k, now);
    var first = r.why[0];
    run.moments.push({ at: run.history.length - 1, turn: run.turn, v: now, n: e.n, later: false, news: e.news, said: said, why: first.who + " " + first.text });
    run.replies++;
    if (first.good) run.cheers++;

    run.last = { k: k, now: now, said: said, developing: !!follow, comments: r.comments, why: r.why };
    checkOver(run);
    return run;
  }

  // pick one outcome using the chances
  function draw(outcomes) {
    var r = Math.random(), sum = 0;
    for (var i = 0; i < outcomes.length; i++) { sum += outcomes[i].p; if (r < sum) return outcomes[i]; }
    return outcomes[outcomes.length - 1];
  }

  // After the result: either a follow-up lands ("later" screen), or the next news arrives.
  // Returns the next screen name: "later", "turn" or "end".
  function advance(run, fromScreen) {
    if (run.over) return "end";
    if (fromScreen === "result") {
      run.turn++;
      var due = null;
      for (var i = 0; i < run.pending.length; i++) if (run.pending[i].due <= run.turn) { due = run.pending[i]; break; }
      if (due) {
        run.pending.splice(run.pending.indexOf(due), 1);
        var out = draw(due.outcomes);
        if (out.v !== 0) {
          run.votes = round1(run.votes + out.v);
          run.history.push(run.votes);
          if (out.v > 0) run.paidOff++; else run.hurt++;
          var e = eventById(due.n);
          run.moments.push({ at: run.history.length - 1, turn: run.turn, v: out.v, n: due.n, later: true, news: "Two turns after: " + e.news, said: due.said, why: out.text });
          checkOver(run);
        }
        run.later = { outcome: out, n: due.n, said: due.said, turn: due.turn };
        return "later";
      }
    }
    drawNext(run);
    return "turn";
  }

  CG.game = { OTHERS: OTHERS, field: field,
    WIN: WIN, LOSE: LOSE, PARTY: PARTY,
    eventById: eventById, newRun: newRun, choose: choose, advance: advance, reactions: reactions
  };
})(window.CG = window.CG || {});
