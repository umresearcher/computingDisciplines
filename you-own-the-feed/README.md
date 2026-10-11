# You Own the Feed

Open `index.html` in a browser (works from a local file or any static host). No build step.

## Structure

```
index.html            page shell: loads the styles and scripts below, renders into #app
css/
  base.css            colour tokens, layout, shared components, phone layout
  screens.css         algorithm bar, feed cards, badges, takeaways, Rule lab
  characters.css      character motion (breathing, blinking, gestures per mood)
  title.css           title page + its motion (hero, floating town, shooting stars)
js/
  data/stories.js     1,889 ReCOVery headlines as window.STORIES
  engine.js           ranking engine (pure functions): FeedEngine.rank / rankLab / metrics
  characters.js       SVG drawings of Morgan, Alex, Jamie, Mia, Leo, Sam -> TS.art
  state.js            game state, decisions -> weights, feeds per scene -> TS.game, TS.util
  ui.js               HTML builders (HUD, bubbles, feed card, ...) -> TS.ui
  screens.js          title + scenes S01-S14 -> TS.screens
  lab.js              Rule lab (bonus screen) -> TS.screens.LAB, TS.lab
  main.js             render loop, clicks and keyboard (Enter, 1, 2)
assets/city.svg       animated title-screen town (cars, walkers, windows, feed screen, bubbles)
tools/
  city.py             regenerates assets/city.svg
  prep_data.py        CSV -> stories (topics, reliability, simulated reactions)
```

## Adding things

- **New screen:** add `SCREENS.S15 = function () { ... }` in `js/screens.js`, link to it with `data-go="S15"`.
- **New character mood:** add the SVG string to `FIG` and the mood to `PEOPLE` in `js/characters.js`.
  Wrap moving parts in `<g class="ch-...">` and give the class keyframes in `css/characters.css`.
- **Town motion:** edit `tools/city.py` (classes + `CSS` block at the end), run
  `python3 tools/city.py assets/city.svg`.
- All motion stops when the visitor's system asks for reduced motion.
