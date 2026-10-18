/* Wiring: keeps the state, draws the current screen, reacts to clicks and keys. */
(function (CG) {
  "use strict";
  var G = CG.game, UI = CG.ui;
  var app = document.getElementById("app");

  var state = {
    screen: "start",          // start, create, turn, result, later, end
    name: "",
    look: { skin: "#E8B48F", hair: "side", hairColor: "#1A1423", glasses: false },
    draftParty: null,         // party picked on the create screen
    run: null,                // the current campaign (see game.js)
    askHome: false,           // the "go to the main screen?" dialog is open
    tab: "news"               // in the feed: news, race (everyone in the race) or profile
  };

  var calm = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var shownScreen = null, shownVotes = null, shownTab = null;   // what the last render showed

  function render() {
    // a new screen or tab plays its entry animation; a redraw of the same one does not
    var newTab = state.screen === shownScreen && state.tab !== shownTab;
    var entering = state.screen !== shownScreen || newTab;
    introStop();
    app.innerHTML = UI[state.screen](state) + (state.askHome ? UI.confirmHome(state) : "");
    app.classList.toggle("enter", entering);
    app.classList.toggle("tab-enter", newTab);
    document.body.className = "on-" + state.screen;
    var votes = state.run ? state.run.votes : null;
    if (votes !== null && shownVotes !== null && votes !== shownVotes) slideVotes(shownVotes, votes);
    shownScreen = state.screen;
    shownTab = state.tab;
    shownVotes = votes;
    if (state.screen === "start") introStart();
    var focus = state.askHome ? app.querySelector(".modal [autofocus]") : app.querySelector("[autofocus]");
    if (focus) focus.focus({ preventScroll: true });
    if (state.screen === "end" && !state.askHome) revealOnScroll();
  }

  // the race bar and the % number glide from the old value to the new one
  function slideVotes(from, to) {
    if (calm) return;
    app.querySelectorAll(".bar-fill").forEach(function (bar) {
      var end = bar.style.width;
      bar.style.transition = "none";
      bar.style.width = UI.barPos(from) + "%";
      void bar.offsetWidth;            // apply the old width before switching the transition back on
      bar.style.transition = "";
      bar.style.width = end;
    });
    var nums = app.querySelectorAll(".vote-num"), t0 = null, ms = 900;
    function step(t) {
      if (t0 === null) t0 = t;
      var k = Math.min(1, (t - t0) / ms), ease = 1 - Math.pow(1 - k, 3);
      nums.forEach(function (n) { n.textContent = UI.pct(from + (to - from) * ease); });
      if (k < 1) requestAnimationFrame(step);
    }
    nums.forEach(function (n) { n.textContent = UI.pct(from); });
    requestAnimationFrame(step);
  }

  /* ---------------------------------------------------------- the intro: a random race, every time
     It starts at 45.0%, moves 5 to 9 times, and ends either at 52%+ (green, "Elected")
     or at 35% or less (red, "Out of the race"). Then it fades and a new race starts. */
  var introTimer = null;
  function introStop() { clearTimeout(introTimer); introTimer = null; }

  function introShow(hero, v) {
    var s = v.toFixed(1);   // "45.0" -> digits 4, 5, 0
    hero.querySelectorAll(".col > span").forEach(function (c, i) {
      c.style.transform = "translate3d(0," + -Number(s[[0, 1, 3][i]]) + "em,0)";
    });
    var arc = Math.max(0, Math.min(1, (v - 30) / 26)) * 180;   // the arc shows 30%..56%
    hero.querySelector(".seats-lit").style.setProperty("--arc", arc.toFixed(1) + "deg");
  }

  function gauss() { return Math.sqrt(-2 * Math.log(1 - Math.random())) * Math.cos(2 * Math.PI * Math.random()); }

  // a random path from 45 to the end value that does not cross 35 or 52 before the last step
  function introPath() {
    var win = Math.random() < 0.55, steps = 5 + Math.floor(Math.random() * 5);
    var end = win ? 52 + Math.random() * 1.6 : 35 - Math.random() * 1.4;
    var walk = [0];
    for (var i = 1; i <= steps; i++) walk.push(walk[i - 1] + gauss() * 2.2);
    var path = [];
    for (var j = 1; j <= steps; j++) {
      var t = j / steps, v = 45 + (end - 45) * t + walk[j] - t * walk[steps];
      if (j < steps) v = Math.max(36, Math.min(51.4, v));
      path.push(Math.round((j === steps ? end : v) * 10) / 10);
    }
    return { win: win, path: path };
  }

  function introStart() {
    var hero = app.querySelector(".intro-hero");
    if (!hero) return;
    var text = hero.querySelector(".live-text");
    hero.classList.add("snap");                       // jump back to 45.0 without animating
    hero.classList.remove("is-win", "is-out");
    text.textContent = "Vote share \u00b7 live";
    introShow(hero, 45);
    void hero.offsetWidth;
    hero.classList.remove("snap", "fading");
    if (calm) return;                                 // motion turned off: a still 45.0%

    var race = introPath(), k = 0;
    function next() {
      introShow(hero, race.path[k++]);
      if (k < race.path.length) { introTimer = setTimeout(next, 1300); return; }
      introTimer = setTimeout(function () {           // the result
        hero.classList.add(race.win ? "is-win" : "is-out");
        text.textContent = race.win ? "Elected" : "Out of the race";
        introTimer = setTimeout(function () {         // fade out, then a new race
          hero.classList.add("fading");
          introTimer = setTimeout(introStart, 700);
        }, 2600);
      }, 900);
    }
    introTimer = setTimeout(next, 1400);
  }

  function go(screen) {
    state.screen = screen;
    state.tab = "news";
    render();
    window.scrollTo(0, 0);
  }

  // end screen: blocks fade in as they scroll into view
  var observer = null;
  function revealOnScroll() {
    var blocks = app.querySelectorAll(".rv");
    if (!("IntersectionObserver" in window)) { blocks.forEach(function (b) { b.classList.add("on"); }); return; }
    if (observer) observer.disconnect();
    observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) { if (en.isIntersecting) { en.target.classList.add("on"); observer.unobserve(en.target); } });
    }, { threshold: 0.15 });
    blocks.forEach(function (b) { observer.observe(b); });
  }

  function act(name) {
    var parts = name.split(":");
    switch (parts[0]) {
      case "to-create": go("create"); break;
      case "look": state.look[parts[1]] = parts[2]; render(); break;
      case "glasses": state.look.glasses = !state.look.glasses; render(); break;
      case "shuffle": state.look = CG.art.randomLook(); render(); break;
      case "party": state.draftParty = parts[1]; render(); break;
      case "launch":
        if (!state.name.trim()) { var input = document.getElementById("name"); if (input) input.focus(); return; }
        if (!state.draftParty) return;
        state.run = G.newRun(state.draftParty);
        go("turn");
        break;
      case "choose":
        if (state.screen !== "turn") return;
        G.choose(state.run, Number(parts[1]));
        go("result");
        break;
      case "next": go(G.advance(state.run, state.screen)); break;
      case "again": state.run = null; go("create"); break;
      case "tab":       // Breaking / Your race / Profile in the feed
        if (!state.run || state.tab === parts[1]) return;
        state.tab = parts[1];
        render();
        window.scrollTo(0, 0);
        break;
      case "story":   // end screen: glide down to the story of the campaign
        var first = app.querySelector(".end .block");
        if (first) first.scrollIntoView({ behavior: calm ? "auto" : "smooth", block: "start" });
        break;
      case "home-ask":   // show the dialog on top, without redrawing the screen under it
        if (state.askHome) return;
        state.askHome = true;
        app.insertAdjacentHTML("beforeend", UI.confirmHome(state));
        app.querySelector(".modal [autofocus]").focus();
        break;
      case "home-no":
        state.askHome = false;
        var m = app.querySelector(".modal-backdrop");
        if (m) m.remove();
        var logoBtn = app.querySelector(".home");
        if (logoBtn) logoBtn.focus({ preventScroll: true });
        break;
      case "home-yes": state.askHome = false; state.run = null; go("start"); break;
    }
  }

  app.addEventListener("click", function (ev) {
    var b = ev.target.closest("[data-act]");
    if (b && b.classList.contains("modal-backdrop") && ev.target.closest(".modal")) return;
    if (b) act(b.getAttribute("data-act"));
  });

  // typing the name: update the preview card without redrawing the input (so the cursor stays put)
  app.addEventListener("input", function (ev) {
    if (ev.target.id !== "name") return;
    state.name = ev.target.value.slice(0, 22);
    var pos = ev.target.selectionStart;
    render();
    var input = document.getElementById("name");
    input.focus();
    input.setSelectionRange(pos, pos);
  });

  document.addEventListener("keydown", function (ev) {
    if (state.askHome) {
      if (ev.key === "Escape") act("home-no");
      return;
    }
    if (ev.target.tagName === "INPUT") {
      if (ev.key === "Enter") act("launch");
      return;
    }
    if (state.screen === "turn" && state.tab === "news" && (ev.key === "1" || ev.key === "2")) act("choose:" + (Number(ev.key) - 1));
  });

  render();
})(window.CG = window.CG || {});
