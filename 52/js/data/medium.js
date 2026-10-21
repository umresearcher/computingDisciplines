/* MEDIUM: policy debates and campaign tools you would hear about in class or in the news.
   The format is described at the top of easy.js. */
window.QUESTIONS = window.QUESTIONS || {};
window.QUESTIONS.medium = [
  { id: "m1", from: "Education Desk", news: "43 million Americans owe money on student loans",
    A: "Cancel up to $10,000 per borrower.", B: "Cut interest rates, but no cancellation.",
    D: [2, 0], R: [-1, 2], I: [0, 2],
    say: [
      ["L", "$10,000 off would let me finally move out of my parents’ house.", "I paid off my loans. Why should I pay for someone else’s? I thought you were fair.", "Nice for them… what about people who didn’t go to college?", "Borrowers were thrilled, but people without loans called it unfair."],
      ["R", "Lower rates help without a giant bill for taxpayers. Smart.", "Lower rates? I’ll still be paying until I’m 50.", "That sounds like a fair middle ground.", "Many saw it as a fair compromise, but borrowers said it barely helps."]
    ] },

  { id: "m2", from: "Housing Report", news: "Rents in big cities rose 20% in two years",
    A: "Cap how much rent can go up each year.", B: "Build more homes by easing zoning rules.",
    D: [2, 1], R: [-1, 2], I: [0, 2],
    say: [
      ["L", "I can’t take another $300 increase. Thank you!", "Rent caps mean landlords stop building. Economists agree on that.", "That would help me this year, for sure.", "Renters were relieved, but landlords and builders warned fewer homes will get built."],
      ["N", "More homes, lower prices. Simple supply and demand.", "Building takes years. I need help with rent now.", "More housing makes sense long-term.", "Many liked a long-term fix, but renters wanted help right now."]
    ] },

  { id: "m3", from: "Poll Desk", news: "New poll of 400 people: you lead by 3 points",
    A: "Celebrate: we are winning!", B: "Say the race is too close to call.",
    D: [1, 1], R: [1, 1], I: [1, 1],
    say: [
      ["N", "We’re winning! Let’s go!", "With 400 people, 3 points is just noise. Don’t get cocky.", "Oh, so you’re ahead? Nice.", "Supporters got excited, but experts noted that a 3-point lead in a small poll is within the margin of error."],
      ["N", "Smart. Keep working like we’re behind.", "Why so gloomy? Enjoy the lead!", "Good, stay humble.", "People liked that you stayed humble and kept working."]
    ],
    followup: [[
      { p: 0.5, v: -3, who: "Poll Desk", text: "A bigger poll of 1,500 people shows you 2 points behind. Your celebration aged badly." },
      { p: 0.5, v: 1, who: "Your field team", text: "The next poll confirmed your lead, and new donors joined." }
    ], null] },

  { id: "m4", from: "Fact Check Desk", news: "Fact-checkers say your ad used a misleading number",
    A: "Fix the ad and say so publicly.", B: "Say the fact-checkers are biased.",
    D: [1, -2], R: [1, 1], I: [1, -2],
    say: [
      ["N", "Owning a mistake? That’s rare. I trust you more now.", "Why admit anything? Now your rival will quote it forever.", "OK, that’s honest. Good.", "Most people respected that you fixed the mistake."],
      ["R", "Fact-checkers always go after our side. Say it!", "You got caught and blamed the referee? I thought you were better than that.", "Hm, sounds like an excuse.", "Some of your supporters agreed, but undecided voters thought it sounded like an excuse."]
    ] },

  { id: "m5", from: "Tech News", news: "A data company offers to sell you detailed profiles of millions of voters",
    A: "Buy them and target ads to each voter.", B: "Say no and use only public data.",
    D: [2, 1], R: [2, 1], I: [3, 1],
    say: [
      ["N", "Smart. Show each voter exactly what they care about.", "You bought my personal data? That’s creepy.", "Wait, how do your ads know so much about me?", "Your ads got sharper, but some voters felt watched."],
      ["N", "Good. My data is not for sale.", "Your rival will buy it anyway. You just fell behind.", "Respect for that.", "People respected the choice, but your ads reached fewer people."]
    ],
    followup: [[
      { p: 0.5, v: -6, who: "Campaign Wire", text: "The data company was hacked. Voters learned your campaign bought their private data." },
      { p: 0.5, v: 0, who: "Your field team", text: "The targeted ads ran quietly. They helped a little, but nobody noticed." }
    ], null] },

  { id: "m6", from: "Viral Video", news: "A fake AI video shows you saying something you never said",
    A: "Post the real video and explain how to spot fakes.", B: "Ignore it so it doesn’t spread further.",
    D: [2, 0], R: [2, 0], I: [2, 0],
    say: [
      ["N", "Thanks for showing the real one. I almost believed the fake.", "Now even more people will go watch the fake.", "Wow, I didn’t know AI could do that.", "People were glad to see the real video and learned how to spot fakes."],
      ["N", "Don’t give it attention. Smart.", "Silence? People will think it’s real.", "So… was that video real or not?", "Some people started to believe the fake because you said nothing."]
    ],
    followup: [null, [
      { p: 0.5, v: -3, who: "Campaign Wire", text: "The fake video reached 10 million views. Some voters still think it’s real." },
      { p: 0.5, v: 0, who: "Campaign Wire", text: "The fake stayed in a few small groups. Most voters never saw it." }
    ]] },

  { id: "m7", from: "Science Desk", news: "Scientists warn this was the hottest year on record",
    A: "Tax carbon pollution and give the money back to families.", B: "Fund new clean technology, with no new taxes.",
    D: [2, 1], R: [-1, 2], I: [1, 1],
    say: [
      ["L", "Polluters pay, families get a check. Love it.", "A new tax on energy? Everything will cost more. I thought you understood that.", "I get a check back? Interesting.", "Climate voters liked it, but many people heard “new tax” and stopped listening."],
      ["R", "Innovation, not taxes. That’s the way.", "That’s too slow. We needed action ten years ago.", "New technology sounds good.", "Many liked the no-tax approach, but climate activists called it too slow."]
    ] },

  { id: "m8", from: "Viral Video", news: "A video of a police arrest goes viral, and protests start",
    A: "Require body cameras and independent investigations.", B: "Stand with the police and fund more training.",
    D: [2, -1], R: [-1, 2], I: [2, 1],
    say: [
      ["L", "Cameras protect everyone, the public and the police.", "You’re treating every officer like a criminal. I expected better.", "Body cameras sound fair to me.", "Protesters welcomed it, but many police officers felt blamed."],
      ["R", "Thank you for backing the people who keep us safe.", "More training? That’s what they always say, and nothing changes.", "Training is good, I guess.", "Police and their families were grateful, but protesters said nothing will change."]
    ] },

  { id: "m9", from: "Tech News", news: "A social media platform bans an account that spread election lies",
    A: "Platforms should remove false election claims.", B: "Platforms shouldn’t decide what’s true.",
    D: [2, -1], R: [-1, 2], I: [0, 1],
    say: [
      ["L", "Lies about voting hurt democracy. Good call.", "So now a tech company decides what I can say? Scary.", "Makes sense for clear lies.", "Many liked fighting election lies, but others worried about censorship."],
      ["R", "Free speech means free speech. Thank you.", "So lies about voting just stay up? I thought you cared about elections.", "Hm, who should decide then?", "Free-speech supporters cheered, but others worried lies will spread."]
    ] },

  { id: "m10", from: "Business News", news: "A company replaces 2,000 call-center workers with AI",
    A: "Make companies that automate pay for retraining.", B: "Let innovation work and cut red tape for new jobs.",
    D: [2, 0], R: [-1, 2], I: [1, 1],
    say: [
      ["L", "If AI takes the job, the company should help the worker.", "Punish companies for new technology? They’ll just move abroad.", "My job could be next. This sounds good.", "Workers liked the protection, but businesses warned it could slow growth."],
      ["R", "New technology always creates new jobs. Let it happen.", "Tell that to the 2,000 people who just lost their jobs.", "I hope new jobs really come.", "Business owners agreed, but worried workers felt left behind."]
    ] },

  { id: "m11", from: "State News", news: "Your state proposes a photo ID requirement to vote",
    A: "Support ID, and make the ID free.", B: "Oppose it: it blocks some legal voters.",
    D: [1, 2], R: [2, -1], I: [2, 0],
    say: [
      ["N", "ID to vote, and free. That’s fair to everyone.", "Even free IDs take time and travel. Some people can’t do that.", "Free ID? That sounds like a fair deal.", "Most voters saw it as a fair compromise, but voting-rights groups were unhappy."],
      ["L", "Millions of citizens don’t have a photo ID. Protect their vote.", "You need ID to buy cold medicine but not to vote? Come on.", "I’m not sure. ID seems normal.", "Voting-rights groups thanked you, but many voters think ID is just common sense."]
    ] },

  { id: "m12", from: "Your Campaign Team", news: "Your team made a harsh but true attack ad about your rival’s tax record",
    A: "Run the attack ad.", B: "Run a positive ad about your plan instead.",
    D: [2, 1], R: [2, 1], I: [1, 2],
    say: [
      ["N", "It’s true! Voters deserve to know.", "Another attack ad. Is this all politics is now?", "Hm, I didn’t know that about your rival.", "The ad hurt your rival, but some voters are tired of negative ads."],
      ["N", "Finally, an ad that tells me what you’ll actually do.", "Nice ad, but nobody remembers positive ads.", "I liked that. It felt hopeful.", "People liked the hopeful tone, but the ad got less attention."]
    ],
    followup: [[
      { p: 0.5, v: 1, who: "Poll Desk", text: "Your rival’s numbers dropped after the ad." },
      { p: 0.5, v: -2, who: "Campaign Wire", text: "Your rival hit back with an even harsher ad. Voters said both campaigns are too negative." }
    ], null] },

  { id: "m13", from: "Campaign Wire", news: "A fringe group known for conspiracy theories endorses you",
    A: "Reject the endorsement publicly.", B: "Say nothing. Votes are votes.",
    D: [2, 1], R: [2, 1], I: [2, 1],
    say: [
      ["N", "Good. Those people don’t speak for us.", "Why push away voters? You need every vote.", "Glad you said no to them.", "Most voters were glad you clearly said no."],
      ["N", "Smart. Don’t make enemies.", "You won’t even reject conspiracy theorists? I thought you had principles.", "Wait, so you’re OK with them?", "Your silence made some voters wonder whether you agree with the group."]
    ],
    followup: [null, [
      { p: 0.6, v: -4, who: "Campaign Wire", text: "Reporters asked you about the group at every event this week. The story wouldn’t die." },
      { p: 0.4, v: 0, who: "Campaign Wire", text: "The endorsement got little attention and faded." }
    ]] },

  { id: "m14", from: "Health News", news: "Health insurance prices jump again this year",
    A: "Offer a public insurance option anyone can buy.", B: "Let people buy insurance from other states.",
    D: [2, -1], R: [-1, 2], I: [2, 0],
    say: [
      ["L", "A cheaper public option? I’d switch tomorrow.", "Government insurance will push private companies out. I thought you knew better.", "More choice sounds good to me.", "Many families liked a cheaper option, but insurers warned about government control."],
      ["R", "More competition across states. Prices will drop.", "Cheaper plans with less coverage? That’s a trap.", "If it lowers prices, sure.", "Some liked more competition, but others worried cheaper plans would cover less."]
    ] },

  { id: "m15", from: "Border Report", news: "Border crossings hit a record this month",
    A: "Hire more immigration judges and create a path to citizenship.", B: "Build more barriers and tighten asylum rules.",
    D: [2, -1], R: [-1, 2], I: [1, 1],
    say: [
      ["L", "Fix the system, don’t just build walls. Agreed.", "A path to citizenship rewards breaking the law. I expected better.", "Faster courts make sense.", "Immigrant communities welcomed it, but many voters said it ignores border security."],
      ["R", "A country needs a border. Thank you.", "Walls don’t fix a broken system. You know that.", "Something has to be done, I suppose.", "Voters worried about the border liked it, but immigrant groups called it cruel."]
    ] },

  { id: "m16", from: "Debate Night", news: "Your rival keeps interrupting you in the debate",
    A: "Stay calm and finish your point.", B: "Interrupt back, louder.",
    D: [2, 1], R: [2, 1], I: [2, -1],
    say: [
      ["N", "You looked like the adult on that stage.", "You let them walk all over you!", "You stayed so calm. I liked that.", "Most viewers thought you looked calm and in control."],
      ["N", "Finally someone fights back!", "Two people yelling at each other. I thought you were more serious than that.", "I couldn’t understand anything either of you said.", "Your fans liked the fight, but many viewers turned off the TV."]
    ] },

  { id: "m17", from: "Factory Town News", news: "A factory town wants protection from cheap imports",
    A: "Put tariffs on imports.", B: "Keep trade open so prices stay low.",
    D: [0, 2], R: [2, 0], I: [2, 0],
    say: [
      ["R", "Protect our jobs from cheap imports. Finally!", "Tariffs are a tax on us, the shoppers. Everything gets more expensive.", "Saving factory jobs sounds good.", "Factory workers cheered, but shoppers worried about higher prices."],
      ["L", "Open trade keeps prices low for everyone.", "Easy to say if your job didn’t move overseas. I thought you cared about us.", "Low prices matter to me.", "Shoppers and economists agreed, but factory workers felt abandoned."]
    ] },

  { id: "m18", from: "Supreme Court News", news: "A court ruling lets each state make its own abortion laws",
    A: "Protect abortion access in national law.", B: "Let each state decide.",
    D: [2, -1], R: [-1, 2], I: [1, 1],
    say: [
      ["L", "This is about women’s health and freedom. Thank you.", "Forcing one rule on every state? That goes too far.", "A big issue. I’m glad you were clear.", "Many women and young voters cheered, but religious voters were upset."],
      ["R", "Each state should decide. That’s how our system works.", "So my rights depend on which state I live in? I expected better from you.", "Letting states decide sounds like a middle path.", "Religious and conservative voters approved, but many women felt their rights were at risk."]
    ] },

  { id: "m19", from: "Your Campaign Team", news: "Only half of young people vote. Your team has money for one turnout push",
    A: "Register voters on college campuses.", B: "Spend it on ads for older voters, who vote more.",
    D: [2, 0], R: [0, 2], I: [2, 0],
    say: [
      ["L", "Someone finally asked me to register!", "Students rarely show up on election day. Wasted money.", "I didn’t know I could register that fast.", "Thousands of students registered, though some doubt they will show up."],
      ["R", "Older voters show up every time. Smart money.", "So young people don’t matter to you?", "Hm, that’s cold, but I get it.", "Older voters liked the attention, but young voters felt ignored."]
    ] },

  { id: "m20", from: "Poll Desk", news: "A third-party candidate is pulling 6% of voters away from you",
    A: "Call them a spoiler who will help your rival win.", B: "Adopt their most popular idea.",
    D: [-1, 2], R: [-1, 2], I: [0, 2],
    say: [
      ["N", "Right. A vote for them is a vote for the other side.", "Attacking a small candidate looks petty and scared.", "Why pick on them?", "Many voters thought attacking a small candidate looked petty."],
      ["N", "Smart. You actually listened to those voters.", "So now you copy other candidates?", "A good idea is a good idea.", "Voters liked that you listened, and some came back to you."]
    ] }
];
