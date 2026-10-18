# The 52

*Reply to the news. Win the vote.*


A short browser game for a political science + computer science class.
You create a candidate, pick a party, and reply to real news headlines from spring 2020.
Every reply moves your votes. Reach 52% and you win; fall to 35% and you drop out.

## Run it

No build step and no install. Either:

- open `index.html` in a browser, or
- serve the folder: `python3 -m http.server` and go to http://localhost:8000

## Files

```
index.html          the page (loads the CSS and JS below)
css/base.css        colors, buttons, create-your-candidate screen
css/intro.css       the intro screen: a random race lights up the arc and ends at 52 (elected) or 35 (out)
css/feed.css        the news feed: posts, replies, comments, the race box, the Your race and Profile tabs
css/end.css         election night: spotlight, chart, scroll-reveal sections
js/data/events.js   the 30 news events + what people comment after each answer
js/game.js          the rules (votes, rival, delayed effects, win/lose) and the other candidates in the race
js/art.js           the logo, the avatar and the full-height candidate drawing (SVG)
js/ui.js            builds the HTML for each screen
js/main.js          state, clicks, keyboard (1 / 2 to answer)
```

## How the game works (the model)

- Start: 45% for Democrat or Republican, 44% for an independent.
- Each turn you pick one of two answers. Its effect for your party is added.
- Then the rival gains 1.1 points (0.85 for an independent).
- Some answers have a `followup`: two turns later one outcome is drawn by chance, and the game shows what happened.
  Example, letting a bot network boost your hashtag: +2 now, then 55% "it trended for three days" (+2) or 45% "researchers traced it to bots" (-4).
  The honest answer is not always the best one, and the risky one is not always punished.
- Win at 52% or more, lose at 35% or less.
- The "Your race" tab shows the whole field. Only your share comes from the game; the rest is filled in around it:
  a Libertarian and a Green candidate with small shares, undecided voters (about 7%, shrinking each turn),
  and your main rival (or both major parties, if you run as an independent) with what is left.
  The names (Elaine Porter, Richard Hale, Wes Calder, Mara Lind) are made up.

With these numbers (simulated, 3,000 games per party):
- a player who weighs each choice wins about 94% of the time;
- always taking the honest answer still wins, but takes longer (about 17 turns instead of 12);
- always taking the risky answer wins 55-75% for Democrats/Republicans and 5% for an independent;
- clicking at random almost always loses.

## Change the content

- **Add or edit a news event:** add an object to `window.EVENTS` in `js/data/events.js`
  (`n`, `news`, `A`, `B`, effects `D`, `R`, `I`, and `tag` + `ps` + `psSrc` for the political science note;
  optional `ctag` + `cs` + `csSrc` for computer science, `src` + `date` for a real headline).
  Every note must be a specific, checkable fact. `psSrc` / `csSrc` are `[label, link]` and show up as a clickable source.
- **Its comments:** add the same `n` to `window.TALK`: two lists (one per answer) of
  `[lean, supporter comment, critic comment, "liked ...", "felt ...", undecided comment, "Undecided voters ..."]`.
  `lean` is `L`, `R` or `N`: who naturally likes this answer.
- **Balance:** change `START`, `DRIFT`, `WIN`, `LOSE` at the top of `js/game.js`.

Headlines marked "Real 2020 headline" come from the ReCOVery COVID-19 news dataset (January to May 2020).
