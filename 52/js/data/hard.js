/* HARD: campaign strategy and how political institutions work. For people who follow politics closely.
   The format is described at the top of easy.js. */
window.QUESTIONS = window.QUESTIONS || {};
window.QUESTIONS.hard = [
  { id: "h1", from: "Your Campaign Team", news: "You have money for one more week of TV ads. Where do you spend it?",
    A: "Three swing states that decide the Electoral College.", B: "Big states where you can boost the national vote.",
    D: [2, 0], R: [2, 0], I: [2, 0],
    say: [
      ["N", "That’s where the election is decided. Good call.", "So voters in the other 47 states don’t matter?", "I’m in a swing state. I see your ads everywhere now.", "Swing-state voters noticed you, though others felt ignored."],
      ["N", "Every vote counts the same. I like that.", "Winning the national vote doesn’t win the presidency. Rookie mistake.", "Nice ads, but my state always votes the same way.", "You gained votes in states you were already going to win or lose."]
    ] },

  { id: "h2", from: "State Capitol", news: "Your party controls the legislature and will redraw the district maps",
    A: "Draw maps that lock in your party’s seats.", B: "Hand the maps to an independent commission.",
    D: [2, 2], R: [2, 2], I: [-1, 2],
    say: [
      ["N", "The other side would do the same. Use the power while you have it.", "That’s gerrymandering. Voters should pick politicians, not the other way around.", "Wait, can you just draw the lines however you want?", "Your party liked the safe seats, but reform groups called it gerrymandering."],
      ["N", "Fair maps. That’s real democracy.", "You just gave away seats your party won fair and square.", "An independent commission sounds fair.", "Most voters liked fair maps, though some party leaders were angry."]
    ],
    followup: [[
      { p: 0.5, v: -4, who: "Campaign Wire", text: "A court threw out the new maps as unfair. Headlines called your party “map riggers.”" },
      { p: 0.5, v: 0, who: "Campaign Wire", text: "The new maps passed with little attention." }
    ], null] },

  { id: "h3", from: "Senate News", news: "Your main bill has 55 votes in the Senate, but a filibuster blocks it",
    A: "End the filibuster to pass it with 51 votes.", B: "Cut a smaller deal that gets 60 votes.",
    D: [2, 2], R: [2, 2], I: [-1, 2],
    say: [
      ["N", "The majority should be able to govern. Do it!", "And when the other party wins the Senate, they’ll use the same rule against you. I thought you thought ahead.", "I don’t really know what a filibuster is, but OK.", "Your base loved the bold move, but many worried the other side will use it next time."],
      ["N", "A deal both parties can live with. Smart.", "Half a bill? You gave away too much.", "Both parties working together? Nice change.", "Moderates liked the deal, but your base said it was too small."]
    ],
    followup: [[
      { p: 0.5, v: -3, who: "Campaign Wire", text: "Ads asked: what will the other party pass with 51 votes? Moderate voters got nervous." },
      { p: 0.5, v: 1, who: "Your field team", text: "The bill passed quickly, and your base turned out in force." }
    ], null] },

  { id: "h4", from: "Supreme Court News", news: "A proposal: 18-year term limits for Supreme Court justices",
    A: "Support term limits.", B: "Keep lifetime appointments.",
    D: [2, 0], R: [-1, 2], I: [2, 0],
    say: [
      ["L", "Nobody should hold that much power for 40 years.", "Lifetime terms keep judges free from politics. I thought you respected the Court.", "18 years sounds like plenty.", "Most voters liked term limits, but legal conservatives warned against changing the Court."],
      ["R", "Lifetime terms protect the Court from politics.", "So one president can shape the Court for 40 years? That’s fine with you?", "Hm, I thought term limits were popular.", "Conservatives agreed, but most voters support term limits."]
    ] },

  { id: "h5", from: "Campaign Finance News", news: "A billionaire’s super PAC wants to spend $50 million on ads for you",
    A: "Welcome the help.", B: "Ask them to stay out and rely on small donors.",
    D: [3, 1], R: [3, 1], I: [2, 2],
    say: [
      ["N", "$50 million! That’s how you win.", "So who do you really work for now? I thought you were for regular people.", "Your ads are everywhere now.", "Your ads flooded the airwaves, but some asked what the billionaire wants in return."],
      ["N", "Small donors keep you honest. Respect.", "Your rival will take the money. You just lost the ad war.", "That’s a principled move.", "People respected the choice, but your rival now has more money for ads."]
    ],
    followup: [[
      { p: 0.5, v: -4, who: "Campaign Wire", text: "Reporters found the billionaire wants a tax break worth $2 billion. Every ad now looks bought." },
      { p: 0.5, v: 0, who: "Campaign Wire", text: "The super PAC ads ran without any scandal." }
    ], null] },

  { id: "h6", from: "Your Campaign Team", news: "Your team found your rival’s 20-year-old drunk driving arrest",
    A: "Leak it to reporters.", B: "Keep the focus on policy.",
    D: [2, 1], R: [2, 1], I: [1, 2],
    say: [
      ["N", "Voters deserve to know who they’re voting for.", "Digging up something from 20 years ago? That’s dirty.", "Hm, that was a long time ago…", "The story hurt your rival, but some voters found it dirty."],
      ["N", "Clean campaign. That’s why I trust you.", "Your rival won’t be this nice to you.", "Good, I care about policy.", "People liked the clean campaign, but the race stayed tight."]
    ],
    followup: [[
      { p: 0.5, v: -3, who: "Campaign Wire", text: "Reporters traced the leak to your campaign. The story became about you, not your rival." },
      { p: 0.5, v: 1, who: "Poll Desk", text: "Your rival’s numbers dropped among older voters." }
    ], null] },

  { id: "h7", from: "Your Campaign Team", news: "It’s time to choose your running mate",
    A: "A popular senator from a swing state.", B: "A loyal ally who excites your base.",
    D: [2, 1], R: [2, 1], I: [2, 0],
    say: [
      ["N", "That pick could win us a whole state!", "A safe, boring choice. Where’s the excitement?", "I know that senator. Good choice.", "Swing-state voters liked the pick, though your base wanted someone more exciting."],
      ["N", "Yes! A real fighter on the ticket!", "Another loyalist? You needed to reach new voters. I thought you knew that.", "Never heard of them.", "Your base got fired up, but swing voters didn’t notice."]
    ] },

  { id: "h8", from: "Breaking News", news: "A week before the election, your campaign’s emails leak. Foreign hackers may be behind it",
    A: "Confirm what’s real and call for an investigation into the hack.", B: "Say all of it is fake.",
    D: [1, 2], R: [1, 2], I: [1, 2],
    say: [
      ["N", "Honest, and you named the real problem: a foreign hack.", "You just confirmed embarrassing emails a week before the vote!", "OK, at least you were straight with us.", "People respected the honesty, and the hack became part of the story."],
      ["N", "Of course it’s fake. They’re trying to steal the election.", "All of it fake? I read them. Some are clearly real. I expected more honesty.", "Hm, so what’s real?", "Your base believed you, but others doubted the denial."]
    ],
    followup: [null, [
      { p: 0.6, v: -5, who: "Fact Check Desk", text: "Experts confirmed several emails are real. Your denial became the bigger story." },
      { p: 0.4, v: 1, who: "Campaign Wire", text: "Nobody could confirm the emails, and the story faded." }
    ]] },

  { id: "h9", from: "Budget Office", news: "Social Security’s trust fund is projected to run short in about ten years",
    A: "Raise the payroll tax on income above $168,000.", B: "Slowly raise the retirement age to 69.",
    D: [3, -1], R: [-1, 1], I: [2, -1],
    say: [
      ["L", "High earners should pay on all their income. Fair.", "That’s a big tax hike on doctors and engineers. I thought you were for growth.", "If it only hits high earners, OK.", "Most voters liked it, but high earners and business groups pushed back."],
      ["R", "People live longer now. The math has to work.", "Work until 69? Try that in construction. I thought you cared about workers.", "I don’t want to work longer…", "Budget experts nodded, but most workers hated working longer."]
    ] },

  { id: "h10", from: "Economy Desk", news: "Inflation is high, so the central bank raises interest rates",
    A: "Respect the Fed’s independence.", B: "Demand the Fed cut rates now.",
    D: [2, 1], R: [2, 1], I: [2, 0],
    say: [
      ["N", "Politicians should keep their hands off the Fed. Good.", "My mortgage just went up and you won’t say anything?", "Sounds responsible.", "Economists and markets liked it, though homebuyers wanted help."],
      ["N", "Yes! High rates are killing small businesses.", "Pressuring the Fed? That’s how inflation gets worse. I expected you to know that.", "Lower rates would help me, honestly.", "Borrowers liked it, but economists warned it could make inflation worse."]
    ],
    followup: [null, [
      { p: 0.5, v: -2, who: "Economy Desk", text: "Markets fell on fears that politicians will control the Fed." },
      { p: 0.5, v: 0, who: "Economy Desk", text: "The Fed ignored the pressure, and markets shrugged." }
    ]] },

  { id: "h11", from: "Tech Policy", news: "A bill would make websites legally responsible for what their users post",
    A: "Support it: platforms must be accountable.", B: "Oppose it: sites would delete anything risky.",
    D: [1, 1], R: [2, 0], I: [0, 2],
    say: [
      ["N", "Big Tech profits from harmful posts. Make them pay.", "Then every site will delete anything that might get it sued. Goodbye free speech.", "Hm, sounds like it’s about time.", "Parents liked holding platforms accountable, but free-speech groups warned of mass deletion."],
      ["N", "Right. Small sites and forums would be sued out of existence.", "So Big Tech gets a free pass again? I thought you’d stand up to them.", "I didn’t know the law protected them like that.", "Tech experts agreed, but many voters wanted Big Tech held accountable."]
    ] },

  { id: "h12", from: "Poll Desk", news: "Two polls disagree. You lead among all registered voters, but not among likely voters",
    A: "Promote the poll where you lead.", B: "Trust the likely-voter poll and push turnout harder.",
    D: [1, 2], R: [1, 2], I: [1, 2],
    say: [
      ["N", "We’re ahead! Share it everywhere.", "The poll that counts is people who actually vote. You know that.", "So you’re winning?", "Supporters were happy, but analysts said the likely-voter poll is the better guide."],
      ["N", "Smart. Polls don’t vote, people do.", "Why highlight the bad poll?", "OK, you’re taking this seriously.", "People liked that you stayed focused on turnout."]
    ],
    followup: [[
      { p: 0.5, v: -3, who: "Poll Desk", text: "Turnout models proved right. You fell behind among people who actually vote." },
      { p: 0.5, v: 0, who: "Poll Desk", text: "Both polls moved closer together. Nothing changed." }
    ], null] },

  { id: "h13", from: "Ballot Measure", news: "Your state will vote on ranked-choice voting",
    A: "Support ranked-choice voting.", B: "Oppose it.",
    D: [0, 1], R: [-1, 2], I: [2, -1],
    say: [
      ["N", "Now I can vote for who I really like without wasting my vote.", "It’s confusing, and counts take weeks. I thought you liked simple elections.", "Ranking candidates? Interesting.", "Reform groups and third-party voters cheered, but some found it confusing."],
      ["R", "One person, one vote. Keep it simple.", "So you like the system that keeps the two parties in charge?", "Hm, ranking did sound complicated.", "Many liked keeping it simple, but reformers said you protect the two-party system."]
    ] },

  { id: "h14", from: "Your Campaign Team", news: "You won the primary by running hard to your side. Now the general election begins",
    A: "Move toward the center on key issues.", B: "Stay exactly where you were.",
    D: [2, 0], R: [2, 0], I: [2, -1],
    say: [
      ["N", "Smart. Now you can win the middle.", "So everything you said in the primary was fake? I thought you believed it.", "You sound more reasonable now.", "Swing voters warmed to you, though some loyal supporters called it a flip-flop."],
      ["N", "Principles! You didn’t change for anyone.", "Great for the primary, but you can’t win the general like that.", "Too extreme for me.", "Your base loved it, but swing voters stayed away."]
    ],
    followup: [[
      { p: 0.4, v: -2, who: "Campaign Wire", text: "Your rival made an ad of your primary quotes next to your new ones: “Which one is real?”" },
      { p: 0.6, v: 1, who: "Poll Desk", text: "Moderate voters began to trust you more." }
    ], null] },

  { id: "h15", from: "Economy Desk", news: "You’re running against the incumbent. The economy is weak",
    A: "Make the election a vote on the economy.", B: "Focus on your personal life story.",
    D: [3, 0], R: [3, 0], I: [2, 0],
    say: [
      ["N", "“Are you better off than four years ago?” Exactly.", "Running on bad news? That’s so negative.", "Yeah, my bills are higher than ever.", "Voters who feel the bad economy agreed with you."],
      ["N", "Your story is so inspiring!", "Nice story, but the economy is on everyone’s mind. I thought you knew that.", "Nice story. But what about prices?", "People liked your story, but most cared more about the economy."]
    ] },

  { id: "h16", from: "Finance Office", news: "A donor wants to give through a shell company so their name stays hidden",
    A: "Refuse: every donor must be public.", B: "Accept: it’s technically legal.",
    D: [1, 2], R: [1, 2], I: [2, 1],
    say: [
      ["N", "Transparency. That’s how it should be.", "You just turned down a lot of money.", "Good, no secret money.", "People respected the transparency."],
      ["N", "Legal is legal. Take it.", "Hidden donors? Who are you hiding? I thought you were for clean politics.", "Hm, why would a donor want to hide?", "Some voters asked who is really funding your campaign."]
    ],
    followup: [null, [
      { p: 0.6, v: -5, who: "Campaign Wire", text: "Reporters traced the shell company to a firm under federal investigation." },
      { p: 0.4, v: 0, who: "Campaign Wire", text: "Nobody looked into the donation." }
    ]] },

  { id: "h17", from: "Tech Desk", news: "Your campaign app leaked 50,000 volunteers’ phone numbers",
    A: "Disclose it today and notify everyone.", B: "Fix it quietly.",
    D: [-1, 0], R: [-1, 0], I: [-1, 0],
    say: [
      ["N", "Telling us right away? That’s why I trust you.", "Embarrassing. Your campaign can’t even run an app.", "At least they told us.", "People appreciated the honesty, though the leak looked sloppy."],
      ["N", "No need to make a big deal out of it.", "Hiding a data leak? That never ends well.", "I didn’t hear about any leak.", "Nobody knows about the leak, yet."]
    ],
    followup: [[
      { p: 0.5, v: 1, who: "Tech Desk", text: "A tech column praised how fast you disclosed the leak. The story faded in a day." },
      { p: 0.5, v: 0, who: "Campaign Wire", text: "The leak got a one-day story and was forgotten." }
    ], [
      { p: 0.5, v: -6, who: "Campaign Wire", text: "Reporters found the leak you hid. The cover-up became the story." },
      { p: 0.5, v: 0, who: "Campaign Wire", text: "The leak never came out." }
    ]] },

  { id: "h18", from: "Weather Service", news: "A hurricane hits a swing state two weeks before the election",
    A: "Pause campaigning and help with relief.", B: "Hold a big rally there to show support.",
    D: [2, -1], R: [2, -1], I: [2, -1],
    say: [
      ["N", "That’s what a leader does. Thank you for helping.", "Two weeks before the vote and you stop campaigning?", "I saw you handing out water. That meant something.", "Voters in the state remembered that you came to help."],
      ["N", "We needed to hear from you!", "A rally while people lose their homes? I thought you had more sense.", "Bad timing…", "Many voters said a rally during a disaster felt tone-deaf."]
    ] },

  { id: "h19", from: "Your Consultants", news: "Polls show voters dislike the other party more than they like yours",
    A: "Run ads about how dangerous the other party is.", B: "Run ads about what you will do.",
    D: [2, 1], R: [2, 1], I: [-1, 2],
    say: [
      ["N", "Yes. People need to see what’s at stake.", "Fear again? Give me a reason to vote FOR you. I expected more.", "These ads make me anxious.", "The fear ads fired up your base, but turned off many swing voters."],
      ["N", "Finally, a plan instead of fear.", "Nice plan, but fear gets people to the polls.", "I actually learned something from that ad.", "Swing voters liked hearing a real plan, though it fired up fewer people."]
    ] },

  { id: "h20", from: "Your Campaign Team", news: "Your last $2 million: knock doors in low-turnout areas or run TV ads for swing voters?",
    A: "Doors: get your supporters to vote.", B: "TV ads: try to change swing voters’ minds.",
    D: [2, 1], R: [2, 1], I: [1, 2],
    say: [
      ["N", "Someone came to my door and reminded me to vote. I went!", "Doors are slow. TV reaches millions.", "A volunteer explained where my polling place is.", "Research suggests personal contact gets supporters to vote, and turnout rose."],
      ["N", "Great ad. Everyone’s talking about it.", "Late ads barely change anyone’s mind. I thought you knew the research.", "I saw the ad. I still don’t know who I’m voting for.", "Your ad got noticed, but very few voters changed their minds."]
    ] }
];
