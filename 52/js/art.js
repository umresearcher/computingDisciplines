/* Drawings: the round avatar (head and shoulders) and the full-height figure for the end screen.
   A "look" is { skin, hair, hairColor, glasses }. The tie is always the party color. */
(function (CG) {
  "use strict";

  var SUIT = "#24324D", LAPEL = "#1A2540", PANTS = "#1C2740", INK = "#1A1423", SHIRT = "#F4F4F4";

  var OPTIONS = {
    skin: [["#F6D3B3", "Light"], ["#E8B48F", "Medium light"], ["#C68B5E", "Medium"], ["#8D5A3B", "Medium dark"], ["#5C3A24", "Dark"]],
    hair: [["short", "Short"], ["side", "Side part"], ["long", "Long"], ["bun", "Bun"], ["curly", "Curly"], ["bald", "Bald"]],
    hairColor: [["#1A1423", "Black"], ["#4A2E1C", "Brown"], ["#8B3A1E", "Auburn"], ["#D9A441", "Blonde"], ["#B8B8BE", "Gray"]]
  };

  // [hair behind the head, hair on top] in a 120 x 120 box, face centered at (60, 54)
  var HAIR = {
    short: ["", "M35 52 C33 31 45 21 60 21 C75 21 87 31 85 52 C81 41 72 35 60 35 C48 35 39 41 35 52 Z"],
    side: ["", "M34 54 C31 30 47 19 63 20 C80 21 89 33 86 54 C83 43 78 37 72 33 C64 41 48 43 36 47 Z"],
    long: ["M30 56 C28 28 44 17 60 17 C77 17 92 28 90 56 L92 104 L28 104 Z", "M35 50 C36 33 46 25 60 25 C74 25 84 33 85 50 C78 40 70 36 60 36 C50 36 42 40 35 50 Z"],
    bun: ["M49 16 a11 11 0 1 0 22 0 a11 11 0 1 0 -22 0 Z", "M36 50 C36 32 46 24 60 24 C74 24 84 32 84 50 C78 40 70 35 60 35 C50 35 42 40 36 50 Z"],
    curly: ["", "M33 54 C27 48 30 38 37 36 C36 28 44 22 51 24 C55 17 66 17 70 23 C78 21 85 28 83 36 C90 39 92 48 87 54 C82 44 72 37 60 37 C48 37 38 44 33 54 Z"],
    bald: ["", ""]
  };

  function randomLook() {
    var pick = function (list) { return list[Math.floor(Math.random() * list.length)][0]; };
    return { skin: pick(OPTIONS.skin), hair: pick(OPTIONS.hair), hairColor: pick(OPTIONS.hairColor), glasses: Math.random() < 0.3 };
  }

  function glasses(y) {
    return '<rect x="43.5" y="' + y + '" width="15" height="12" rx="4" fill="#fff" fill-opacity=".15" stroke="' + INK + '" stroke-width="2.4"/>' +
      '<rect x="61.5" y="' + y + '" width="15" height="12" rx="4" fill="#fff" fill-opacity=".15" stroke="' + INK + '" stroke-width="2.4"/>' +
      '<path d="M58.5 ' + (y + 5) + ' L61.5 ' + (y + 5) + '" stroke="' + INK + '" stroke-width="2.4"/>';
  }

  // the face, in the 120 x 120 box. mood: "happy" (avatar), "cheer" (open smile), "sad"
  function face(look, mood) {
    var h = HAIR[look.hair] || HAIR.side, brow = look.hair === "bald" ? "#3A2A20" : look.hairColor;
    var s = '<circle cx="37" cy="56" r="5" fill="' + look.skin + '"/><circle cx="83" cy="56" r="5" fill="' + look.skin + '"/>' +
      '<ellipse cx="60" cy="54" rx="23" ry="26" fill="' + look.skin + '"/>' +
      (h[1] ? '<path d="' + h[1] + '" fill="' + look.hairColor + '"/>' : "");
    if (mood === "sad") {
      s += '<path d="M46 50 Q51 47.5 56 45.5" stroke="' + brow + '" stroke-width="2.6" fill="none" stroke-linecap="round"/>' +
        '<path d="M64 45.5 Q69 47.5 74 50" stroke="' + brow + '" stroke-width="2.6" fill="none" stroke-linecap="round"/>' +
        '<path d="M47.5 58 Q51 60.5 54.5 58" stroke="' + INK + '" stroke-width="2.4" fill="none" stroke-linecap="round"/>' +
        '<path d="M65.5 58 Q69 60.5 72.5 58" stroke="' + INK + '" stroke-width="2.4" fill="none" stroke-linecap="round"/>' +
        '<path d="M53 70 Q60 64.5 67 70" stroke="' + INK + '" stroke-width="2.6" fill="none" stroke-linecap="round"/>';
      if (look.glasses) s += glasses(51);
    } else {
      s += '<path d="M46 48.5 Q51 45.5 56 48.5" stroke="' + brow + '" stroke-width="2.6" fill="none" stroke-linecap="round"/>' +
        '<path d="M64 48.5 Q69 45.5 74 48.5" stroke="' + brow + '" stroke-width="2.6" fill="none" stroke-linecap="round"/>' +
        '<circle cx="51" cy="56" r="2.8" fill="' + INK + '"/><circle cx="69" cy="56" r="2.8" fill="' + INK + '"/>' +
        (mood === "cheer"
          ? '<path d="M50 64 Q60 77 70 64 Z" fill="' + INK + '"/><path d="M53 64.5 L67 64.5 L65.5 67.5 L54.5 67.5 Z" fill="#fff"/>'
          : '<path d="M52 66 Q60 72.5 68 66" stroke="' + INK + '" stroke-width="2.6" fill="none" stroke-linecap="round"/>');
      if (look.glasses) s += glasses(50);
    }
    return s;
  }

  // Round avatar. bg is the circle color.
  function avatar(look, tie, bg, size) {
    var h = HAIR[look.hair] || HAIR.side;
    return '<span class="av" style="width:' + size + 'px;height:' + size + 'px;background:' + bg + '">' +
      '<svg viewBox="0 0 120 120" width="100%" height="100%" aria-hidden="true">' +
      (h[0] ? '<path d="' + h[0] + '" fill="' + look.hairColor + '"/>' : "") +
      '<rect x="52" y="70" width="16" height="20" rx="4" fill="' + look.skin + '"/>' +
      '<path d="M10 122 C12 96 34 86 60 86 C86 86 108 96 110 122 Z" fill="' + SUIT + '"/>' +
      '<path d="M48.5 86 L60 113 L71.5 86 Z" fill="' + SHIRT + '"/>' +
      '<path d="M56.5 88 L63.5 88 L62 93.5 L58 93.5 Z" fill="' + tie + '"/>' +
      '<path d="M58 93.5 L62 93.5 L64.5 111 L60 117 L55.5 111 Z" fill="' + tie + '"/>' +
      '<path d="M48.5 86 L58.5 111 L45 99 L43 89 Z" fill="' + LAPEL + '"/><path d="M71.5 86 L61.5 111 L75 99 L77 89 Z" fill="' + LAPEL + '"/>' +
      '<circle cx="80" cy="100" r="2.2" fill="' + tie + '"/>' +
      face(look, "happy") + "</svg></span>";
  }

  // Full-height figure (300 x 540 box). mood "win": arm raised. mood "lose": head down, open briefcase.
  function figure(look, tie, mood, height) {
    var h = HAIR[look.hair] || HAIR.side, sk = look.skin;
    var back = h[0] ? '<path d="' + h[0].replace("L92 104 L28 104", "L91 84 Q60 96 29 84") + '" fill="' + look.hairColor + '"/>' : "";
    var legs = function (top) {
      return '<path d="M112 ' + top + ' L148 ' + top + ' L146 490 L116 490 Z" fill="' + PANTS + '"/><path d="M152 ' + top + ' L188 ' + top + ' L184 490 L154 490 Z" fill="' + PANTS + '"/>' +
        '<ellipse cx="128" cy="496" rx="24" ry="9" fill="#0B0B0F" stroke="#2A2A33" stroke-width="2"/><ellipse cx="172" cy="496" rx="24" ry="9" fill="#0B0B0F" stroke="#2A2A33" stroke-width="2"/>';
    };
    var torso = function (dy) {
      var t = function (v) { return v + dy; };
      return '<rect x="138" y="' + t(110) + '" width="24" height="36" rx="6" fill="' + sk + '"/>' +
        '<path d="M92 ' + t(160) + ' C92 ' + t(146) + ' 112 ' + t(138) + ' 150 ' + t(138) + ' C188 ' + t(138) + ' 208 ' + t(146) + ' 208 ' + t(160) + ' L200 ' + t(336) + ' L100 ' + t(336) + ' Z" fill="' + SUIT + '"/>' +
        '<path d="M130 ' + t(140) + ' L150 ' + t(214) + ' L170 ' + t(140) + ' Z" fill="' + SHIRT + '"/>' +
        '<path d="M144 ' + t(142) + ' L156 ' + t(142) + ' L154 ' + t(153) + ' L146 ' + t(153) + ' Z" fill="' + tie + '"/>' +
        '<path d="M146 ' + t(153) + ' L154 ' + t(153) + ' L159 ' + t(206) + ' L150 ' + t(218) + ' L141 ' + t(206) + ' Z" fill="' + tie + '"/>' +
        '<path d="M130 ' + t(140) + ' L148 ' + t(210) + ' L122 ' + t(178) + ' L117 ' + t(150) + ' Z" fill="' + LAPEL + '"/>' +
        '<path d="M170 ' + t(140) + ' L152 ' + t(210) + ' L178 ' + t(178) + ' L183 ' + t(150) + ' Z" fill="' + LAPEL + '"/>' +
        '<circle cx="188" cy="' + t(170) + '" r="4.5" fill="' + tie + '"/>' +
        '<path d="M150 ' + t(228) + ' L150 ' + t(336) + '" stroke="' + LAPEL + '" stroke-width="2"/>' +
        '<circle cx="150" cy="' + t(240) + '" r="3.2" fill="' + LAPEL + '"/><circle cx="150" cy="' + t(266) + '" r="3.2" fill="' + LAPEL + '"/>';
    };
    var arm = function (d, hx, hy) { return '<path d="' + d + '" stroke="' + SUIT + '" stroke-width="30" fill="none" stroke-linecap="round"/><circle cx="' + hx + '" cy="' + hy + '" r="14" fill="' + sk + '"/>'; };
    var svg;
    if (mood === "win") {
      svg = legs(328) + arm("M200 166 C212 220 214 262 210 300", 209, 310) + torso(0) +
        '<g class="fig-wave"><path d="M100 166 C84 130 76 96 70 64" stroke="' + SUIT + '" stroke-width="30" fill="none" stroke-linecap="round"/>' +
        '<path d="M62 70 L80 66" stroke="' + SHIRT + '" stroke-width="7" stroke-linecap="round"/><circle cx="69" cy="50" r="17" fill="' + sk + '"/></g>' +
        '<g transform="translate(150 84) scale(1.47) translate(-60 -54)">' + back + face(look, "cheer") + "</g>";
    } else {
      svg = '<g class="fig-sad">' +
        '<rect class="paper p1" x="236" y="262" width="24" height="31" rx="2" fill="#E9E6DD"/>' +
        '<rect class="paper p2" x="214" y="244" width="22" height="29" rx="2" fill="#F4F2EC"/>' +
        '<rect class="paper p3" x="256" y="240" width="22" height="28" rx="2" fill="#DCD8CC"/>' +
        legs(336) + arm("M100 176 C86 228 84 268 90 306", 91, 316) + torso(12) +
        '<path d="M200 176 C214 230 218 270 214 308" stroke="' + SUIT + '" stroke-width="30" fill="none" stroke-linecap="round"/>' +
        '<path d="M200 320 C200 306 228 306 228 320" stroke="#3B2616" stroke-width="5" fill="none"/>' +
        '<rect x="184" y="320" width="76" height="54" rx="6" fill="#6B4630"/>' +
        '<path d="M184 320 L260 320 L272 290 L196 294 Z" fill="#7E5639"/><path d="M190 322 L254 322 L262 300 L200 302 Z" fill="#3B2616"/>' +
        '<rect x="216" y="338" width="12" height="7" rx="2" fill="#C9A54C"/>' +
        '<circle cx="214" cy="318" r="14" fill="' + sk + '"/>' +
        '<g transform="translate(150 100) rotate(-9) scale(1.47) translate(-60 -54)">' + back + face(look, "sad") + "</g></g>";
    }
    var w = Math.round(height * 300 / 540);
    return '<svg class="figure" viewBox="0 0 300 540" width="' + w + '" height="' + height + '" role="img" aria-label="' +
      (mood === "win" ? "Your candidate, arm raised in victory" : "Your candidate, head down, holding an open briefcase") + '">' + svg + "</svg>";
  }

  // The 52 logo: a light tile with a big "52" and a small "THE" in the corner.
  // Below 48px the "THE" is dropped so the number stays readable.
  function logo(size) {
    var small = size < 48;
    return '<svg class="logo" width="' + size + '" height="' + size + '" viewBox="0 0 100 100" role="img" aria-label="The 52">' +
      '<rect width="100" height="100" rx="22" fill="#E7E9EA"/>' +
      (small
        ? '<text x="50" y="68" text-anchor="middle" font-family="Figtree, system-ui, sans-serif" font-weight="900" font-size="54" letter-spacing="-3" fill="#000">52</text>'
        : '<text x="12" y="24" font-family="Figtree, system-ui, sans-serif" font-weight="800" font-size="10.5" letter-spacing="3.2" fill="#000">THE</text>' +
          '<text x="91" y="89" text-anchor="end" font-family="Figtree, system-ui, sans-serif" font-weight="900" font-size="66" letter-spacing="-4" fill="#000">52</text>') +
      "</svg>";
  }
  var LOGO = logo(28);

  CG.art = { OPTIONS: OPTIONS, randomLook: randomLook, avatar: avatar, figure: figure, logo: logo, LOGO: LOGO };
})(window.CG = window.CG || {});
