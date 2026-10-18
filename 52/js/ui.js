/* Screens. Each function returns HTML for one screen; main.js puts it on the page.
   Buttons carry data-act="..." and main.js decides what each one does. */
(function (CG) {
  "use strict";
  var G = CG.game, A = CG.art;

  var OUTLET = { "Chicago Sun-Times": "CST", "Reuters": "R", "NPR": "NPR", "The Verge": "V", "Health Impact News": "HIN" };
  var WIRE = "Campaign Wire";

  function esc(s) { return String(s).replace(/[&<>"']/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]; }); }
  function sign(x) { return x > 0 ? "+" + x : x < 0 ? "−" + Math.abs(x) : "0"; }
  function tone(x) { return x > 0 ? "up" : x < 0 ? "down" : "flat"; }
  // a source link that opens in a new tab: src = [label, url]
  function source(src) {
    return src ? ' <a class="src" href="' + esc(src[1]) + '" target="_blank" rel="noopener">' + esc(src[0]) + " \u2197</a>" : "";
  }
  function pct(v) { return v.toFixed(1) + "%"; }
  function round1(x) { return Math.round(x * 10) / 10; }
  function barPos(v) { return Math.max(0, Math.min(100, (v - 30) / 26 * 100)); } // the bar shows 30%..56%
  function myName(st) { return st.name.trim() || "You"; }
  function handle(st) { return "@" + (myName(st).toLowerCase().replace(/[^a-z0-9]/g, "") || "you") + "2020"; }
  function me(st, size) { var p = G.PARTY[st.run.party]; return A.avatar(st.look, p.color, p.dark, size); }

  var ICON = {
    news: '<svg width="24" height="24" viewBox="0 0 24 24" aria-hidden="true"><path d="M3 11 L12 3 L21 11 V21 H14 V15 H10 V21 H3 Z" fill="currentColor"/></svg>',
    profile: '<svg width="24" height="24" viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="8" r="4.2" stroke="currentColor" stroke-width="2.2" fill="none"/><path d="M4 21 C4.5 16 8 14 12 14 C16 14 19.5 16 20 21" stroke="currentColor" stroke-width="2.2" fill="none" stroke-linecap="round"/></svg>',
    race: '<svg width="24" height="24" viewBox="0 0 24 24" aria-hidden="true"><path d="M4 20 V12 M10 20 V6 M16 20 V10 M22 20 V3" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"/></svg>',
    reply: '<svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true"><path d="M4 5 H20 V16 H11 L6 20 V16 H4 Z" stroke="currentColor" stroke-width="2" fill="none" stroke-linejoin="round"/></svg>',
    share: '<svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true"><path d="M7 4 L3 8 L7 12 M3 8 H17 V12 M17 20 L21 16 L17 12 M21 16 H7 V12" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg>',
    like: '<svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true"><path d="M12 20 C5 15 3 11.5 3 8.5 A4.5 4.5 0 0 1 12 6.5 A4.5 4.5 0 0 1 21 8.5 C21 11.5 19 15 12 20 Z" stroke="currentColor" stroke-width="2" fill="none"/></svg>',
    down: '<svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true"><path d="M6 9 L12 15 L18 9" stroke="currentColor" stroke-width="2.4" fill="none" stroke-linecap="round"/></svg>',
    star: '<svg width="30" height="30" viewBox="0 0 24 24" aria-hidden="true"><path d="M12 1.5 L14.9 8.4 L22.4 9 L16.7 13.9 L18.4 21.2 L12 17.3 L5.6 21.2 L7.3 13.9 L1.6 9 L9.1 8.4 Z" fill="currentColor"/></svg>'
  };

  /* ------------------------------------------------------------ 1. start */
  // 100 dots in a half circle (6 rows); the lit ones show the vote share between 30% and 56%
  var SEATS = (function () {
    var rows = [200, 224, 248, 272, 296, 320], total = rows.reduce(function (a, r) { return a + r; }, 0), out = "", used = 0;
    rows.forEach(function (r, i) {
      var n = i === rows.length - 1 ? 100 - used : Math.round(100 * r / total);
      used += n;
      for (var j = 0; j < n; j++) {
        var a = Math.PI * (1 - j / (n - 1));
        out += '<circle cx="' + (340 + r * Math.cos(a)).toFixed(1) + '" cy="' + (340 - r * Math.sin(a)).toFixed(1) + '" r="8.5"/>';
      }
    });
    return out;
  })();

  function start() {
    var seats = function (cls) { return '<svg class="' + cls + '" viewBox="0 0 680 350" aria-hidden="true"><g>' + SEATS + "</g></svg>"; };
    var roll = function (c) {
      var d = ""; for (var i = 0; i < 10; i++) d += "<i>" + i + "</i>";
      return '<span class="col"><span class="c' + c + '">' + d + "</span></span>";
    };
    // the arc is zoomed to 30%..56% (the same range as the race bar in the game), so 35 sits far left and 52 far right
    var mark = function (share, cls, label) {
      var ang = Math.PI * (1 - (share - 30) / 26);
      var px = function (r) { return (340 + r * Math.cos(ang)).toFixed(1); }, py = function (r) { return (340 - r * Math.sin(ang)).toFixed(1); };
      return '<g class="' + cls + '"><line x1="' + px(186) + '" y1="' + py(186) + '" x2="' + px(338) + '" y2="' + py(338) + '"/>' +
        '<text x="' + px(352) + '" y="' + (+py(352) - 6).toFixed(1) + '" text-anchor="middle">' + label + "</text></g>";
    };
    var line = '<svg class="seat-line" viewBox="0 0 680 350" aria-hidden="true">' + mark(35, "out", "35 OUT") + mark(52, "win", "52 WIN") + "</svg>";
    var records = [["To win", "52%", "up"], ["You\u2019re out", "35%", "down"], ["News stories", "30", ""], ["Random players who win", "4%", ""]]
      .map(function (r) { return '<li><span>' + r[0] + '</span><b class="' + r[2] + '">' + r[1] + "</b></li>"; }).join("");
    return '<main class="intro">' +
      '<header class="intro-brand">' + A.LOGO + "<b>The 52</b></header>" +
      '<section class="intro-hero">' +
      '<div class="seats-wrap">' + seats("seats-off") + '<div class="seats-on">' + seats("seats-lit") + "</div>" + line +
      '<div class="count" role="img" aria-label="Vote share counter, rising toward 52 percent">' + roll(0) + roll(1) + '<i class="pt"></i>' + roll(2) + '<span class="pc">%</span></div></div>' +
      '<p class="live"><i></i><span class="live-text">Vote share \u00b7 live</span></p>' +
      "<h1>Run the campaign.<br>Reach 52.</h1>" +
      '<button class="btn big-btn" data-act="to-create">Start your campaign</button>' +
      "</section>" +
      '<ul class="records">' + records + "</ul>" +
      "</main>";
  }

  /* ------------------------------------------------------------ 2. create your candidate */
  function create(st) {
    var L = st.look, dp = st.draftParty ? G.PARTY[st.draftParty] : null, name = st.name.trim(), ready = !!(name && st.draftParty);
    var swatches = function (key, labelSuffix) {
      return A.OPTIONS[key].map(function (o) {
        return '<button class="swatch" data-act="look:' + key + ":" + o[0] + '" aria-label="' + o[1] + " " + labelSuffix + '" aria-pressed="' + (L[key] === o[0]) + '" style="background:' + o[0] + '"></button>';
      }).join("");
    };
    var hair = A.OPTIONS.hair.map(function (o) {
      return '<button class="chip" data-act="look:hair:' + o[0] + '" aria-pressed="' + (L.hair === o[0]) + '">' + o[1] + "</button>";
    }).join("");
    var party = ["D", "R"].map(function (k) {
      var p = G.PARTY[k];
      return '<button class="party" data-act="party:' + k + '" aria-pressed="' + (st.draftParty === k) + '" style="--pc:' + p.color + '"><i style="background:' + p.color + '"></i><span><b>' + p.name + "</b><small>start 45%</small></span></button>";
    }).join("");
    var hint = ready ? "Your rival is already campaigning." : !name ? "Type a name to continue." : "Pick a party to continue.";
    return '<div class="plain"><header class="topbar">' + '<button class="home" data-act="home-ask" aria-label="The 52: go to the main screen">' + A.LOGO + "<b>The 52</b></button>" + "</header>" +
      '<main class="create">' +
      '<section class="create-form"><h1>Create your candidate</h1>' +
      '<label class="field"><span>Name</span><input id="name" maxlength="22" autocomplete="off" placeholder="Type your candidate’s name" value="' + esc(st.name) + '"></label>' +
      '<div class="field"><div class="field-head"><span>Look</span><button class="chip small" data-act="shuffle">Shuffle</button></div>' +
      '<div class="row"><span class="row-label">Skin</span><div class="row-items">' + swatches("skin", "skin") + "</div></div>" +
      '<div class="row"><span class="row-label">Hair</span><div class="row-items">' + hair + "</div></div>" +
      '<div class="row"><span class="row-label">Hair color</span><div class="row-items">' + swatches("hairColor", "hair") + "</div></div>" +
      '<div class="row"><span class="row-label">Glasses</span><div class="row-items"><button class="chip" data-act="glasses" aria-pressed="' + !!L.glasses + '">' + (L.glasses ? "On" : "Off") + "</button></div></div></div>" +
      '<div class="field"><span>Party</span><div class="parties">' + party + "</div>" +
      '<button class="party-ind" data-act="party:I" aria-pressed="' + (st.draftParty === "I") + '">Prefer not to choose <small>· run as an independent, start 44%</small></button></div>' +
      "</section>" +
      '<aside class="create-preview"><div class="card" style="--pc:' + (dp ? dp.color : "#536471") + '">' +
      '<div class="card-stripe"></div><div class="card-body">' +
      A.avatar(L, dp ? dp.color : "#536471", dp ? dp.dark : "#2F3336", 200) +
      '<b class="card-name' + (name ? "" : " empty") + '">' + esc(name || "Your name") + "</b>" +
      '<span class="muted">for President · 2020</span>' +
      '<span class="card-party">' + (dp ? dp.name + " · start " + (st.draftParty === "I" ? 44 : 45) + "%" : "No party yet") + "</span>" +
      "</div></div>" +
      '<button class="btn wide" data-act="launch" aria-disabled="' + !ready + '">Launch campaign</button>' +
      '<p class="muted center" role="status">' + hint + "</p></aside>" +
      "</main></div>";
  }

  /* ------------------------------------------------------------ the feed shell (left menu, feed, right rail) */
  function race(run) {
    var p = G.PARTY[run.party];
    return '<div class="race"><div class="race-top"><b class="race-num vote-num">' + pct(run.votes) + '</b><span class="muted">of the vote</span></div>' +
      '<div class="bar"><div class="bar-track"><div class="bar-fill" style="width:' + barPos(run.votes) + "%;background:" + p.color + '"></div></div>' +
      '<i class="tick out" style="left:19.2%"></i><i class="tick win" style="left:84.6%"></i>' +
      '<span class="tick-label" style="left:19.2%">35 out</span><span class="tick-label" style="left:84.6%">52 wins</span></div>' +
      '<span class="muted small">Your rival gains ' + run.drift + " points each turn.</span></div>";
  }

  function shell(st, feedHtml) {
    var run = st.run, p = G.PARTY[run.party];
    var upNext = run.deck.slice(-2).reverse().map(function (n) {
      var e = G.eventById(n);
      return '<li><span class="muted small">' + esc(e.src || WIRE) + "</span><b>" + esc(e.news) + "</b></li>";
    }).join("");
    var developing = run.pending.length ? '<li><span class="small warn">Still developing</span><b>Something you posted is getting attention.</b></li>' : "";
    var nav = function (key, icon, label) {
      return '<button class="nav-item' + (st.tab === key ? " on" : "") + '" data-act="tab:' + key + '"' + (st.tab === key ? ' aria-current="page"' : "") + ">" + ICON[icon] + label + "</button>";
    };
    var view = st.tab === "race" ? raceView(st) : st.tab === "profile" ? profileView(st) : feedHtml;
    return '<div class="shell">' +
      '<nav class="side" aria-label="Menu"><div class="brand">' + '<button class="home" data-act="home-ask" aria-label="The 52: go to the main screen">' + A.LOGO + "<b>The 52</b></button>" + "</div>" +
      nav("news", "news", "News") + nav("race", "race", "Your race") + nav("profile", "profile", "Profile") +
      '<button class="me" data-act="tab:profile" aria-label="Open your profile">' + me(st, 44) + '<div><b>' + esc(myName(st)) + '</b><span class="muted small">' + p.name + " · turn " + run.turn + "</span></div></button></nav>" +
      '<main class="feed" id="feed">' +
      '<div class="race-mini"><button class="home" data-act="home-ask" aria-label="The 52: go to the main screen">' + A.logo(30) + "</button>" + '<button class="mini-me" data-act="tab:profile" aria-label="Open your profile">' + me(st, 32) + "</button>" + '<b class="vote-num">' + pct(run.votes) + '</b><div class="bar-track"><div class="bar-fill" style="width:' + barPos(run.votes) + "%;background:" + p.color + '"></div><i style="left:19.2%" class="mt out"></i><i style="left:84.6%" class="mt win"></i></div><span class="muted small">T' + run.turn + "</span></div>" +
      tabs(st) + view + "</main>" +
      '<aside class="rail"><section class="box"><h2>Your race</h2>' + race(run) + "</section>" +
      '<section class="box"><h2>Also breaking</h2><ul class="list">' + upNext + developing + "</ul></section></aside>" +
      "</div>";
  }

  function newsPost(e, threaded) {
    var src = e.src || WIRE, ini = OUTLET[src] || "W";
    var chips = e.low ? '<span class="pill bad">Low-reliability site</span>' : e.src ? '<span class="pill">Real 2020 headline</span>' : "";
    // threaded = the same news is already on screen (after you reply), so it should not animate in again
    return '<article class="post' + (threaded ? " stay" : "") + '">' +
      '<div class="post-side"><span class="outlet' + (e.low ? " low" : e.src ? "" : " wire") + '">' + ini + "</span>" + (threaded ? '<i class="thread-line"></i>' : "") + "</div>" +
      '<div class="post-body"><div class="post-head"><b>' + esc(src) + '</b> <span class="muted">· ' + esc(e.date || "this week") + "</span></div>" +
      '<h1 class="headline">' + esc(e.news) + "</h1>" + (chips ? '<div class="chips">' + chips + "</div>" : "") +
      '<div class="actions" aria-hidden="true"><span>' + ICON.reply + "Reply</span><span>" + ICON.share + "Share</span><span>" + ICON.like + "Like</span></div></div></article>";
  }

  function tabs(st) {
    var color = G.PARTY[st.run.party].color;
    var one = function (key, label) {
      var on = st.tab === key;
      return '<button class="tab' + (on ? " on" : "") + '" data-act="tab:' + key + '"' + (on ? ' aria-current="page"' : "") + '><span' + (on ? ' style="border-color:' + color + '"' : "") + ">" + label + "</span></button>";
    };
    return '<div class="tabs">' + one("news", "Breaking") + one("race", "Your race") + one("profile", "Profile") + "</div>";
  }

  /* ------------------------------------------------------------ the "Your race" tab: everyone in the race */
  function raceView(st) {
    var run = st.run, p = G.PARTY[run.party], F = G.field(run), O = G.OTHERS;
    var rows = [{ key: "me", name: myName(st), party: p.name, color: p.color, av: me(st, 44), you: true }];
    ["D", "R", "L", "G"].forEach(function (k) {
      if (F.now[k] === undefined) return;
      var c = O[k];
      rows.push({ key: k, name: c.name, party: c.party, color: c.color, av: A.avatar(c.look, c.color, c.dark, 44) });
    });
    rows.forEach(function (r) { r.v = F.now[r.key]; r.d = round1(r.v - F.before[r.key]); });
    rows.sort(function (a, b) { return b.v - a.v; });

    var row = function (r, i) {
      var change = r.d === 0 ? "no change" : (r.d > 0 ? "\u25B2 " : "\u25BC ") + Math.abs(r.d).toFixed(1);
      return '<li class="cand' + (r.you ? " you" : "") + '" style="--c:' + r.color + ";--i:" + i + '">' +
        '<span class="cand-rank">' + (i + 1) + "</span>" + r.av +
        '<div class="cand-main"><div class="cand-head"><b>' + esc(r.name) + "</b>" + (r.you ? '<span class="you-tag">You</span>' : "") +
        (i === 0 ? '<span class="lead-tag">Leading</span>' : "") + '<span class="muted">' + r.party + "</span></div>" +
        '<div class="cand-bar"><i style="width:' + (r.v / 60 * 100).toFixed(1) + '%"></i><em></em></div></div>' +
        '<div class="cand-num"><b>' + pct(r.v) + '</b><span class="' + tone(r.d) + '">' + change + "</span></div></li>";
    };
    var undecided = '<li class="cand undecided" style="--i:' + rows.length + '"><span class="cand-rank"></span><span class="q">?</span>' +
      '<div class="cand-main"><div class="cand-head"><b>Undecided</b><span class="muted">still choosing</span></div>' +
      '<div class="cand-bar"><i style="width:' + (F.now.undecided / 60 * 100).toFixed(1) + '%"></i></div></div>' +
      '<div class="cand-num"><b>' + pct(F.now.undecided) + "</b></div></li>";

    // one line that says where you stand
    var best = rows.filter(function (r) { return !r.you; })[0], gap = round1(run.votes - best.v);
    var standing = gap > 0 ? "You lead " + esc(best.name) + " by " + gap.toFixed(1) + " points."
      : gap < 0 ? "You trail " + esc(best.name) + " by " + Math.abs(gap).toFixed(1) + " points." : "You are tied with " + esc(best.name) + ".";
    standing += " You need " + (run.votes >= 52 ? "nothing more" : (52 - run.votes).toFixed(1) + " more to reach 52%") + ".";

    return '<section class="view race-view">' +
      '<header class="view-head"><h2>The race</h2><p class="soft">' + standing + "</p></header>" +
      '<ol class="cands">' + rows.map(row).join("") + undecided + "</ol>" +
      '<p class="muted small legend"><i></i>52% wins outright \u00b7 35% and you\u2019re out</p>' +
      raceChart(run, p, F) +
      "</section>";
  }

  // you vs. your main rival, turn by turn
  function raceChart(run, p, F) {
    var mine = run.history, theirs = F.rivalHistory, rc = G.OTHERS[F.rivalKey];
    var all = mine.concat(theirs), lo = Math.min(30, Math.floor(Math.min.apply(null, all)) - 2), hi = Math.max(56, Math.ceil(Math.max.apply(null, all)) + 2);
    var n = Math.max(1, mine.length - 1), W = 440, H = 190;
    var x = function (i) { return (20 + i * (W - 120) / n).toFixed(1); };
    var y = function (v) { return (12 + (hi - v) / (hi - lo) * (H - 24)).toFixed(1); };
    var line = function (vals, color, w) {
      var pts = vals.map(function (v, i) { return x(i) + "," + y(v); }).join(" ");
      var last = vals.length - 1;
      return '<polyline points="' + pts + '" fill="none" stroke="' + color + '" stroke-width="' + w + '" stroke-linejoin="round" stroke-linecap="round"/>' +
        '<circle cx="' + x(last) + '" cy="' + y(vals[last]) + '" r="5" fill="' + color + '"/>';
    };
    var tag = function (vals, color, label) {
      var last = vals.length - 1;
      return '<text x="' + (+x(last) + 12) + '" y="' + (+y(vals[last]) + 5) + '" fill="' + color + '" font-size="14" font-weight="700">' + esc(label) + "</text>";
    };
    var guide = function (v, color, label) {
      return '<line x1="20" x2="' + (W - 20) + '" y1="' + y(v) + '" y2="' + y(v) + '" stroke="' + color + '" stroke-dasharray="5 6" stroke-width="1.5"/>' +
        '<text x="20" y="' + (+y(v) - 6) + '" fill="' + color + '" font-size="12" font-weight="700">' + label + "</text>";
    };
    return '<div class="race-chart"><h3>You vs. ' + esc(rc.name) + "</h3>" +
      '<svg viewBox="0 0 ' + W + " " + H + '" role="img" aria-label="Your vote share and your main rival\u2019s, turn by turn">' +
      guide(52, "#34D399", "52 WIN") + guide(35, "#FB7185", "35 OUT") +
      line(theirs, rc.color, 2.5) + line(mine, p.color, 3.5) +
      tag(mine, p.color, "You") + tag(theirs, rc.color, rc.name.split(" ")[1]) + "</svg></div>";
  }

  /* ------------------------------------------------------------ the profile tab */
  function profileView(st) {
    var run = st.run, p = G.PARTY[run.party];
    var posts = run.moments.slice().reverse().map(function (m) {
      return '<article class="p-post"><div class="post-head"><span class="muted small">Turn ' + m.turn + (m.later ? " \u00b7 two turns later" : "") + "</span>" +
        '<span class="delta ' + tone(m.v) + '">' + sign(m.v) + " votes</span></div>" +
        '<span class="muted small">' + esc(m.later ? m.news.replace("Two turns after: ", "") : m.news) + "</span>" +
        "<p>" + (m.later ? esc(m.why) : "\u201C" + esc(m.said) + "\u201D") + "</p></article>";
    }).join("");
    var stat = function (v, label) { return "<span><b>" + v + "</b> " + label + "</span>"; };
    return '<section class="view profile">' +
      '<div class="p-banner" style="background:linear-gradient(120deg,' + p.dark + "," + p.color + ')"></div>' +
      '<div class="p-head">' + me(st, 112) + '<span class="card-party" style="--pc:' + p.color + '">' + p.name + "</span></div>" +
      '<div class="p-id"><h2>' + esc(myName(st)) + '</h2><span class="muted">' + handle(st) + "</span>" +
      '<p class="soft">Running for president \u00b7 2020 \u00b7 started at ' + pct(run.history[0]) + "</p></div>" +
      '<div class="p-stats">' + stat(pct(run.votes), "vote share") + stat(run.replies, "replies") + stat(run.cheers, "base cheers") + "<span>turn <b>" + run.turn + "</b></span>" + "</div>" +
      '<h3 class="p-sub">Your replies</h3>' +
      (posts || '<p class="muted p-empty">No replies yet. Answer the news to start your campaign.</p>') +
      "</section>";
  }

  /* ------------------------------------------------------------ 3. a turn */
  function turn(st) {
    var run = st.run, e = G.eventById(run.current), p = G.PARTY[run.party];
    var feed = newsPost(e, false) +
      '<div class="composer">' + me(st, 44) + '<div class="composer-body"><span class="muted big">Reply as ' + esc(myName(st)) + "…</span>" +
      '<button class="answer" data-act="choose:0"><span class="key">1</span>' + esc(e.A) + "</button>" +
      '<button class="answer" data-act="choose:1"><span class="key">2</span>' + esc(e.B) + "</button>" +
      '<span class="muted small">or press 1 / 2</span></div></div>';
    return shell(st, feed);
  }

  /* ------------------------------------------------------------ 4. after you reply */
  function result(st) {
    var run = st.run, e = G.eventById(run.current), p = G.PARTY[run.party], last = run.last;
    var comments = last.comments.map(function (c) {
      var cp = c.party ? G.PARTY[c.party] : null;
      return '<article class="comment"><span class="initial" style="background:' + (cp ? cp.dark : "#333639") + ";color:" + (cp ? cp.light : "#C4C7CA") + '">' + c.name.charAt(0) + "</span>" +
        '<div><div><b>' + c.name + '</b> <span class="muted">· ' + c.tag + "</span></div><p>" + esc(c.text) + "</p></div></article>";
    }).join("");
    var why = last.why.map(function (w) {
      return '<p><b class="' + (w.good === true ? "up" : w.good === false ? "down" : "") + '">' + w.who + "</b> " + esc(w.text) + "</p>";
    }).join("");
    var feed = newsPost(e, true) +
      '<div class="result" aria-live="polite">' +
      '<article class="post mine">' + '<div class="post-side">' + me(st, 44) + "</div>" +
      '<div class="post-body"><div class="post-head"><b>' + esc(myName(st)) + '</b> <span class="muted">' + handle(st) + ' · now</span><span class="delta ' + tone(last.now) + '">' + sign(last.now) + " votes</span></div>" +
      '<p class="said">' + esc(last.said) + "</p></div></article>" +
      comments +
      '<div class="why"><h2>What people think</h2>' + why +
      '<div class="learn"><span class="muted">Learn more</span><p><b class="ps">' + esc(e.tag) + ":</b> " + esc(e.ps) + source(e.psSrc) + "</p>" +
      (e.cs ? '<p><b class="cs">' + esc(e.ctag) + ":</b> " + esc(e.cs) + source(e.csSrc) + "</p>" : "") + "</div>" +
      (last.developing ? '<p class="warn">This story is still developing.</p>' : "") +
      '<div class="why-foot"><span class="muted small">Rival −' + run.drift + ' this turn</span><button class="btn" data-act="next" autofocus>' + (run.over ? "See the results" : "Next news") + "</button></div>" +
      "</div></div>";
    return shell(st, feed);
  }

  /* ------------------------------------------------------------ 5. something comes back two turns later */
  function later(st) {
    var run = st.run, L = run.later, o = L.outcome;
    var ini = o.who === "Fact Check Desk" ? "FC" : o.who === "Your field team" ? "FT" : "W";
    var badge = o.v > 0 ? "calm" : o.v < 0 ? "alarm" : "wire";
    var feed = '<div class="feed-title"><h2>Two turns later\u2026</h2></div>' +
      '<article class="post later-post"><div class="post-side"><span class="outlet ' + badge + '">' + ini + "</span></div>" +
      '<div class="post-body"><div class="post-head"><b>' + esc(o.who) + '</b> <span class="muted">\u00b7 now</span><span class="delta ' + tone(o.v) + '">' + sign(o.v) + " votes</span></div>" +
      '<h1 class="headline">' + esc(o.text) + "</h1>" +
      '<div class="quote"><div class="post-head">' + me(st, 24) + "<b>" + esc(myName(st)) + '</b> <span class="muted">\u00b7 turn ' + L.turn + "</span></div><p>" + esc(L.said) + "</p></div>" +
      '<div class="why-foot"><span></span><button class="btn" data-act="next" autofocus>' + (run.over ? "See the results" : "Next news") + "</button></div></div></article>";
    return shell(st, feed);
  }

  /* ------------------------------------------------------------ 6. election night */
  function chart(run, color, marks) {
    var h = run.history, n = Math.max(1, h.length - 1);
    var x = function (i) { return 48 + i * 900 / n; };
    var y = function (v) { return 14 + (56 - Math.max(30, Math.min(56, v))) / 26 * 196; };
    var pts = h.map(function (v, i) { return x(i).toFixed(1) + "," + y(v).toFixed(1); }).join(" ");
    var dots = marks.map(function (m) {
      var i = Math.min(m.at, n), yy = y(h[i]), c = m.v >= 0 ? "#34D399" : "#FB7185";
      return '<g class="mk"><circle cx="' + x(i).toFixed(1) + '" cy="' + yy.toFixed(1) + '" r="9" fill="' + c + '" stroke="#000" stroke-width="4"/>' +
        '<text x="' + x(i).toFixed(1) + '" y="' + (m.v >= 0 ? yy - 16 : yy + 30).toFixed(1) + '" fill="' + c + '" font-size="18" font-weight="800" text-anchor="middle">' + sign(m.v) + "</text></g>";
    }).join("");
    return '<svg class="chart" viewBox="0 0 1000 260" role="img" aria-label="Your votes from turn 1 to turn ' + run.turn + '">' +
      '<rect x="48" y="14" width="900" height="30.2" fill="#34D399" fill-opacity=".08"/><rect x="48" y="172.3" width="900" height="37.7" fill="#FB7185" fill-opacity=".08"/>' +
      '<line x1="48" x2="948" y1="44.2" y2="44.2" stroke="#34D399" stroke-dasharray="6 6"/><line x1="48" x2="948" y1="172.3" y2="172.3" stroke="#FB7185" stroke-dasharray="6 6"/>' +
      '<text x="40" y="49" fill="#34D399" font-size="15" text-anchor="end">52</text><text x="40" y="177" fill="#FB7185" font-size="15" text-anchor="end">35</text>' +
      '<polygon points="' + x(0).toFixed(1) + ",236 " + pts + " " + x(n).toFixed(1) + ',236" fill="' + color + '" fill-opacity=".10"/>' +
      '<polyline class="ln" points="' + pts + '" fill="none" stroke="' + color + '" stroke-width="5" stroke-linejoin="round" stroke-linecap="round"/>' + dots +
      '<text x="48" y="254" fill="#71767B" font-size="15">turn 1</text><text x="948" y="254" fill="#71767B" font-size="15" text-anchor="end">turn ' + run.turn + "</text></svg>";
  }

  function end(st) {
    var run = st.run, p = G.PARTY[run.party], win = run.over === "win";
    // the three moments that moved votes the most, shown in the order they happened
    var top = run.moments.slice().sort(function (a, b) { return Math.abs(b.v) - Math.abs(a.v) || b.at - a.at; }).slice(0, 3)
      .sort(function (a, b) { return a.at - b.at; });
    var lessons = [], used = [];
    top.forEach(function (m) {   // one lesson per news story
      if (used.indexOf(m.n) >= 0) return;
      used.push(m.n);
      var e = G.eventById(m.n);
      lessons.push(m.later && e.cs ? [e.cs, e.ctag, e.csSrc] : [e.ps, e.tag, e.psSrc]);
    });
    var confetti = win ? '<div class="confetti" aria-hidden="true">' + [18, 30, 44, 61, 74, 86].map(function (left, i) {
      return '<i style="left:' + left + "%;animation-delay:" + (i * 0.7) + "s;background:" + (i % 3 === 0 ? "#FACC15" : i % 3 === 1 ? p.color : "#E7E9EA") + '"></i>';
    }).join("") + "</div>" : "";
    var cards = top.map(function (m, i) {
      return '<div class="rv moment" style="transition-delay:' + (i * 0.18) + 's"><span class="big-delta ' + tone(m.v) + '">' + sign(m.v) + "</span>" +
        '<span class="muted small">Turn ' + m.turn + " · " + esc(m.news) + "</span><b>“" + esc(m.said) + "”</b><span class=\"soft\">" + esc(m.why) + "</span></div>";
    }).join("");
    var numbers = [[run.replies, "replies posted"], [run.cheers, "times your base cheered"], [run.paidOff, "paid off later"], [run.hurt, "came back to hurt you"]].map(function (q, i) {
      return '<div class="rv number" style="transition-delay:' + (i * 0.12) + "s;border-color:" + p.color + '"><span>' + q[0] + "</span><small>" + q[1] + "</small></div>";
    }).join("");
    var lessonRows = lessons.map(function (l, i) {
      return '<div class="rv lesson" style="transition-delay:' + (i * 0.15) + 's"><span class="lesson-n" style="color:' + p.color + '">0' + (i + 1) + "</span><span class=\"lesson-t\">" + esc(l[0]) + '<span class="lesson-src">Source:' + source(l[2]) + '</span></span><span class="pill">' + esc(l[1]) + "</span></div>";
    }).join("");
    return '<div class="end' + (win ? " is-win" : " is-lose") + '">' +
      '<section class="hero"><header class="topbar">' + '<button class="home" data-act="home-ask" aria-label="The 52: go to the main screen">' + A.LOGO + "<b>The 52</b></button>" + "</header>" +
      '<div class="beam" aria-hidden="true"></div>' + confetti +
      '<div class="hero-body"><span class="kicker">ELECTION NIGHT · NOVEMBER 3, 2020</span>' +
      '<div class="stage">' + A.figure(st.look, p.color, win ? "win" : "lose", 430) +
      (win ? '<div class="podium" style="border-top-color:' + p.color + '">' + ICON.star + "</div>" : '<div class="floor"></div>') + "</div>" +
      '<h1 class="end-name">' + esc(myName(st)) + "</h1>" +
      '<p class="end-line"><b class="' + (win ? "up" : "down") + '">' + (win ? "President-elect" : "Dropped out of the race") + '</b><span class="sep">|</span><span class="soft">' + p.name + " · " + pct(run.votes) + "</span></p>" +
      '<div class="end-actions"><button class="btn" data-act="again">Play again</button>' +
      '<button class="btn ghost" data-act="story">How your campaign went' + ICON.down + "</button></div></div></section>" +
      '<section class="rv block"><span class="eyebrow">01 · THE RACE</span><h2>From ' + run.history[0] + "% to " + pct(run.votes) + "</h2><div class=\"chart-wrap\">" + chart(run, p.color, top) + "</div></section>" +
      '<section class="block"><div class="rv"><span class="eyebrow">02 · TURNING POINTS</span><h2>Three moments that decided it</h2></div><div class="moments">' + cards + "</div></section>" +
      '<section class="block"><div class="rv"><span class="eyebrow">03 · IN NUMBERS</span><h2>Your campaign</h2></div><div class="numbers">' + numbers + "</div></section>" +
      '<section class="block"><div class="rv"><span class="eyebrow">04 · WHAT IT SHOWED</span><h2>What your campaign showed</h2></div>' + lessonRows + "</section>" +
      '<section class="rv again"><h2>Run again?</h2><p class="soft">Same news. Different choices. Different result. Try another party.</p><button class="btn" data-act="again">Start a new campaign</button></section>' +
      "</div>";
  }

  /* ------------------------------------------------------------ confirm: back to the main screen */
  function confirmHome(st) {
    var playing = st.run && st.screen !== "end";
    return '<div class="modal-backdrop" data-act="home-no">' +
      '<div class="modal" role="dialog" aria-modal="true" aria-labelledby="home-title" aria-describedby="home-text">' +
      A.logo(56) + '<h2 id="home-title">Go to the main screen?</h2>' +
      '<p id="home-text" class="soft">' + (playing ? "Your campaign will end, and this run will be lost." : "You can come back and start again any time.") + "</p>" +
      '<div class="modal-actions"><button class="btn ghost" data-act="home-no" autofocus>No, stay</button>' +
      '<button class="btn" data-act="home-yes">Yes, go</button></div></div></div>';
  }

  CG.ui = { barPos: barPos, pct: pct, confirmHome: confirmHome, start: start, create: create, turn: turn, result: result, later: later, end: end };
})(window.CG = window.CG || {});
