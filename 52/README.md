# The 52

*Reply to the news. Win the vote.*


A short browser game for a political science + computer science class.
You create a candidate, pick a party and a difficulty, and answer the questions a campaign runs into.
Every answer moves your votes. Reach 52% and you win; fall to 35% and you drop out.

## Run it

No build step and no install. Either:

- open `index.html` in a browser, or
- serve the folder: `python3 -m http.server` and go to http://localhost:8000


or https://umresearcher.github.io/computingDisciplines/52/

## Files

```
index.html          the page (loads the CSS and JS below)
css/base.css        colors, buttons, create-your-candidate screen (name, look, party, difficulty)
css/intro.css       the intro screen: a random race lights up the arc and ends at 52 (elected) or 35 (out)
css/feed.css        the news feed: posts, replies, comments, the race box, the Your race and Profile tabs
css/end.css         election night: spotlight, chart, scroll-reveal sections
js/data/easy.js     20 questions: everyday issues, no politics knowledge needed (for example, a school student)
js/data/medium.js   20 questions: policy debates from class and the news (for example, a college student)
js/data/hard.js     20 questions: campaign strategy and institutions (for people who follow politics closely)
js/game.js          the rules (votes, rival, delayed effects, win/lose, election night) and the other candidates
js/art.js           the logo, the avatar and the full-height candidate drawing (SVG)
js/ui.js            builds the HTML for each screen
js/main.js          state, clicks, keyboard (1 / 2 to answer)
```

## How the game works (the model)

- On the create screen you pick a difficulty: Easy, Medium or Hard. It only changes which 20 questions you get; the rules are the same.
- The questions come in random order, and each one comes up once per campaign.
- Each turn you pick one of two answers. Its effect for your party is added, then your rival gains 1.1 points.
- After each answer, three people comment (your supporter, an undecided voter and, on party issues, a voter from the other side),
  and one sentence sums up how people reacted.
- Some answers have a `followup`: two turns later one outcome is drawn by chance, and the game shows what happened.
  The safe answer is not always the best one, and the risky one is not always punished.
- A campaign ends in one of three ways:
  - you reach the win line (you win);
  - you fall to the out line (you drop out);
  - the 20 questions run out: election night. You win if you have more votes than your main rival.
- Democrat or Republican: start at 45%, win at 52%, out at 35%. You face one main rival.
- Independent: a three-way race. You start at 30%, level with both major candidates, and must beat both of them:
  win at 38%, out at 22%.
- The "Your race" tab shows the whole field. Only your share comes from the game; the rest is filled in around it:
  a Libertarian and a Green candidate with small shares, undecided voters (about 7%, shrinking each turn),
  and your main rival (or both major candidates, if you run as an independent) with what is left.
  The names (Elaine Porter, Richard Hale, Wes Calder, Mara Lind) are made up.

With these numbers (simulated, 4,000 games per party and level):
- a player who picks the answer with the best expected result always wins: in about 9-12 questions as a Democrat or Republican,
  11-17 as an independent;
- a player who only looks at the immediate effect wins every time on Easy and Medium, but only about 2 in 3 times
  on Hard as a Democrat or Republican: there you need to think about what happens next;
- clicking at random wins only 3-18% of the time.

## Change the content

- **Add or edit a question:** add an object to the list in `js/data/easy.js`, `medium.js` or `hard.js`.
  The format is described at the top of `easy.js`: `id`, `from`, `news`, answers `A` and `B`,
  effects `D`, `R`, `I` (`[after A, after B]`), `say` (comments and the one-sentence summary for each answer)
  and an optional `followup`. Every question needs effects for all three parties.
- **Comments:** `say` has one list per answer: `[lean, fan, critic, undecided, summary]`.
  `lean` is `L`, `R` or `N`: who naturally likes this answer (with `N`, the vote change decides).
- **Balance:** change `START`, `DRIFT`, `WIN`, `LOSE` at the top of `js/game.js`.
- **A new difficulty:** add a data file that sets `window.QUESTIONS.<name>`, load it in `index.html`,
  and add it to `LEVELS` in `js/game.js`.
