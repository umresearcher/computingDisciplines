/* Runtime: renders the current screen into #app and handles clicks and keys (Enter, 1, 2). */
(function (TS) {
  "use strict";
  var E = window.FeedEngine;
  var state = TS.game.state, reset = TS.game.reset;
  var labRank = TS.lab.labRank, labStart = TS.lab.labStart, LAB_DIALS = TS.lab.LAB_DIALS, WORDS = TS.lab.WORDS, LAB_DAYS = TS.lab.LAB_DAYS;

  var SCREENS = TS.screens;
  var app = document.getElementById("app");
  function render(focus) {
    var fn = SCREENS[state.screen] || SCREENS.title;
    app.innerHTML = state.screen === "title" ? fn() : '<div class="screen">' + fn() + "</div>";
    if (focus !== false) {
      window.scrollTo(0, 0);
      var main = app.querySelector("main");
      if (main) main.focus({ preventScroll: true });
    }
  }
  function go(id) { state.screen = id; render(); }
  function refocus(sel) { var el = app.querySelector(sel); if (el) el.focus(); }

  function act(a) {
    var p = a.split(":");
    if (p[0] === "rule1") { state.rule1 = p[1]; go("S05"); }
    else if (p[0] === "rule2") { state.rule2 = p[1]; go("S08"); }
    else if (p[0] === "rule3") { state.rule3 = p[1]; go("S11"); }
    else if (p[0] === "check1" || p[0] === "check2") {
      var pick = Number(p[1]);
      state[p[0] + "Pick"] = pick;
      state[p[0]] = pick === 2 ? "right" : "wrong";
      render(false); refocus("[data-enter]");
    }
    else if (p[0] === "restart") { reset(); go("title"); }
    else if (p[0] === "dial" || p[0] === "labreset" || p[0] === "laboff") {
      var L = state.lab;
      L.prev = labRank(L).slice(0, 3).map(function (c) { return c.s.id; });
      if (p[0] === "dial") {
        L.w = Object.assign({}, L.w); L.w[p[1]] = Number(p[2]);
        L.change = LAB_DIALS.filter(function (d) { return d.key === p[1]; })[0].label + " \u2192 " + WORDS[Number(p[2])];
      } else if (p[0] === "labreset") { L.w = labStart(); L.change = "Back to your rules"; }
      else { L.w = { wR: 0, wI: 0, wE: 0, wV: 0, wH: 0 }; L.change = "All dials off"; }
      render(false); refocus('[data-act="' + a + '"]');
    }
    else if (p[0] === "reader") { state.lab.reader = p[1]; state.lab.prev = null; state.lab.change = "Now showing " + E.READERS[p[1]].name + "\u2019s feed"; render(false); refocus('[data-act="' + a + '"]'); }
    else if (p[0] === "day") { state.lab.day = p[1]; state.lab.prev = null; state.lab.change = "Switched to the " + LAB_DAYS.filter(function (d) { return d[0] === p[1]; })[0][1].toLowerCase(); render(false); refocus('[data-act="' + a + '"]'); }
  }

  app.addEventListener("click", function (e) {
    var el = e.target.closest("[data-go],[data-act]");
    if (!el || el.disabled) return;
    if (el.hasAttribute("data-act")) act(el.getAttribute("data-act"));
    else go(el.getAttribute("data-go"));
  });
  document.addEventListener("keydown", function (e) {
    if (e.altKey || e.ctrlKey || e.metaKey) return;
    var el = null, ae = document.activeElement;
    if (e.key === "1" || e.key === "2") el = app.querySelector('[data-key="' + e.key + '"]:not([disabled])');
    else if (e.key === "Enter" && (!ae || ae === document.body || ae.tagName === "MAIN")) el = app.querySelector("[data-enter]");
    if (el) { e.preventDefault(); el.click(); }
  });

  function start(saved) {
    if (saved && saved.screen && SCREENS[saved.screen]) reset(saved);
    render();
  }
  var hot = window.claude && window.claude.hot;
  if (hot && hot.snapshot) hot.snapshot(function () { return state; });
  if (hot && hot.ready) hot.ready(start); else start(hot && hot.data);


})(window.TS = window.TS || {});
