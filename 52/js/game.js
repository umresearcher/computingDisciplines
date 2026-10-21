/* Game rules. No drawing here: this file only changes numbers.
   One turn = a question arrives -> you pick answer 0 or 1 -> votes move -> the rival gains a little.
   A campaign ends at the win line, at the out line, or when the questions run out (election night).
   A Democrat or Republican faces one main rival: win at 52%, out at 35%.
   An independent starts level with BOTH major candidates (a three-way race), so its lines are lower. */
(function (CG) {
  "use strict";

  var START = { D: 45, R: 45, I: 30 };      // starting votes, %
  var DRIFT = { D: 1.1, R: 1.1, I: 1.1 };  // the rival gains this much every turn
  var WIN = { D: 52, R: 52, I: 38 }, LOSE = { D: 35, R: 35, I: 22 };
  // the lines and the range the race bar and the charts show for this party
  function goal(party) { return { win: WIN[party], lose: LOSE[party], lo: LOSE[party] - 5, hi: WIN[party] + 4 }; }
  var DELAY = 2;
  var DAYS = 100;                            // the campaign lasts this many days; each question takes an equal share                             // follow-ups land this many turns later

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

  // the three difficulty levels; each has its own set of questions in js/data/<level>.js
  var LEVELS = [
    { key: "easy", name: "Easy", hint: "Everyday issues. No politics knowledge needed." },
    { key: "medium", name: "Medium", hint: "Policy debates you hear in class or in the news." },
    { key: "hard", name: "Hard", hint: "Campaign strategy and how institutions work." }
  ];

  function questionById(id) {
    for (var level in window.QUESTIONS) {
      var list = window.QUESTIONS[level];
      for (var i = 0; i < list.length; i++) if (list[i].id === id) return list[i];
    }
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

  // every question of the level, in random order. Each one comes up once per campaign.
  function newRun(party, level) {
    var deck = shuffle(window.QUESTIONS[level].map(function (q) { return q.id; }));
    var run = {
      party: party, level: level, votes: START[party], turn: 1, drift: DRIFT[party],
      deck: deck, total: deck.length, current: null,
      pending: [],       // follow-ups waiting to land
      history: [START[party]],
      moments: [],       // every vote change, for the end screen
      cheers: 0, paidOff: 0, hurt: 0, replies: 0,
      last: null, later: null, over: null, endedBy: null, final: null
    };
    run.current = run.deck.pop();
    return run;
  }

  // which question this is (1-based) out of how many
  function progress(run) { return { n: run.total - run.deck.length, of: run.total }; }

  // the countdown to Election Day. `left` = days left while you answer the current question,
  // `afterLast` = days left once it is answered (0 after the last question)
  function days(run) {
    var per = DAYS / run.total, n = progress(run).n;
    return { total: DAYS, left: Math.round(DAYS - (n - 1) * per), afterLast: Math.round(DAYS - n * per) };
  }

  // Who comments and what they think. `k` is the answer (0 or 1), `now` the vote change.
  function reactions(party, q, k, now) {
    var T = q.say[k], lean = T[0];
    var people = { D: ["Dana", "Democratic voter"], R: ["Ray", "Republican voter"] };
    var likes = function (p) { return lean === "L" ? p === "D" : p === "R"; };
    // on party-coded answers a D or R base reacts to the lean; otherwise to the result
    var baseLikes = (party !== "I" && lean !== "N") ? likes(party) : now >= 0;
    var comments = [];

    var base = party === "I" ? { name: "Sky", tag: "your supporter", party: "I" } : { name: people[party][0], tag: "your base", party: party };
    comments.push({ name: base.name, tag: base.tag, party: base.party, text: baseLikes ? T[1] : T[2] });
    comments.push({ name: "Chris", tag: "undecided", party: null, text: T[3] });
    if (lean !== "N") {
      var other = party === "I" ? (lean === "L" ? "R" : "D") : (party === "D" ? "R" : "D");
      comments.push({ name: people[other][0], tag: party === "I" ? people[other][1] : "rival supporter", party: other, text: likes(other) ? T[1] : T[2] });
    }
    return { comments: comments, summary: T[4], baseLikes: baseLikes };
  }

  function checkOver(run) {
    if (run.votes >= WIN[run.party]) { run.over = "win"; run.endedBy = "votes"; }
    else if (run.votes <= LOSE[run.party]) { run.over = "lose"; run.endedBy = "votes"; }
  }

  // no questions left: election night. You win if you have more votes than your main rival
  // (for an independent: more than each of the two major candidates).
  function election(run) {
    var F = field(run), rival = F.now[F.rivalKey];
    run.over = run.votes > rival ? "win" : "lose";
    run.endedBy = "election";
    run.final = { rivalKey: F.rivalKey, rival: rival };
  }

  // The player picks answer k for the current question.
  function choose(run, k) {
    var q = questionById(run.current);
    var now = q[run.party][k];
    var said = k === 0 ? q.A : q.B;
    var follow = q.followup && q.followup[k];

    if (follow) run.pending.push({ due: run.turn + DELAY, outcomes: follow, id: q.id, said: said, turn: run.turn });

    run.votes = round1(run.votes + now - run.drift);
    run.history.push(run.votes);

    var r = reactions(run.party, q, k, now);
    run.moments.push({ at: run.history.length - 1, turn: run.turn, v: now, id: q.id, later: false, news: q.news, said: said, why: r.summary });
    run.replies++;
    if (r.baseLikes) run.cheers++;

    run.last = { k: k, now: now, said: said, developing: !!follow, comments: r.comments, summary: r.summary };
    checkOver(run);
    return run;
  }

  // pick one outcome using the chances
  function draw(outcomes) {
    var r = Math.random(), sum = 0;
    for (var i = 0; i < outcomes.length; i++) { sum += outcomes[i].p; if (r < sum) return outcomes[i]; }
    return outcomes[outcomes.length - 1];
  }

  function dueNext(run) {
    for (var i = 0; i < run.pending.length; i++) if (run.pending[i].due <= run.turn + 1) return run.pending[i];
    return null;
  }

  // true when the next click leads to the end screen (for the button label)
  function endsNext(run, fromScreen) {
    return !!run.over || (!run.deck.length && (fromScreen !== "result" || !dueNext(run)));
  }

  // After the result: either a follow-up lands ("later" screen), or the next question arrives,
  // or there are no questions left and the election is held. Returns "later", "turn" or "end".
  function advance(run, fromScreen) {
    if (run.over) return "end";
    if (fromScreen === "result") {
      var due = dueNext(run);
      if (due || run.deck.length) run.turn++;
      if (due) {
        run.pending.splice(run.pending.indexOf(due), 1);
        var out = draw(due.outcomes);
        if (out.v !== 0) {
          run.votes = round1(run.votes + out.v);
          run.history.push(run.votes);
          if (out.v > 0) run.paidOff++; else run.hurt++;
          var q = questionById(due.id);
          run.moments.push({ at: run.history.length - 1, turn: run.turn, v: out.v, id: due.id, later: true, news: q.news, said: due.said, why: out.text });
          checkOver(run);
        }
        run.later = { outcome: out, id: due.id, said: due.said, turn: due.turn };
        return "later";
      }
    }
    if (!run.deck.length) { election(run); return "end"; }
    run.current = run.deck.pop();
    return "turn";
  }

  CG.game = { OTHERS: OTHERS, field: field, LEVELS: LEVELS,
    WIN: WIN, LOSE: LOSE, goal: goal, PARTY: PARTY, START: START,
    questionById: questionById, newRun: newRun, choose: choose, advance: advance, endsNext: endsNext, progress: progress, days: days
  };
})(window.CG = window.CG || {});
