/* EASY: everyday issues. You do not need to follow politics to answer these.
   Format of one question (the same in easy.js, medium.js and hard.js):
     id      unique id ("e1", "m1", "h1", ...)
     from    who posts the news (shown as the account name)
     news    the headline
     A, B    the two answers
     D, R, I vote change for a Democrat, Republican and Independent: [after A, after B]
     say     what people say after each answer: [lean, fan, critic, undecided, summary]
               lean: L = left-coded answer, R = right-coded, N = neither (then the vote change decides who likes it)
               fan = a voter who likes the answer, critic = one who doesn't, undecided = a voter in the middle,
               summary = one sentence: how people reacted overall
     followup (optional) what can happen two turns later: [after A, after B], each a list of { p: chance, v: votes, who, text } */
window.QUESTIONS = window.QUESTIONS || {};
window.QUESTIONS.easy = [
  { id: "e1", from: "School Board", news: "Should school lunch be free for every student?",
    A: "Free lunch for every kid.", B: "Free only for families who need it.",
    D: [2, 0], R: [-1, 2], I: [1, 2],
    say: [
      ["L", "No kid should learn on an empty stomach. Thank you!", "Free for everyone? Even for rich families? I thought you were smarter with money.", "My kids would love that. But who pays?", "Parents loved the idea, but many taxpayers asked who will pay for it."],
      ["R", "Help the families who need it. That’s fair.", "So the poor kids get singled out in the lunch line? Not cool.", "Sounds reasonable to me.", "Most people found it fair, but some said kids feel embarrassed when only some eat for free."]
    ] },

  { id: "e2", from: "Local News", news: "Fast-food workers go on strike for a higher minimum wage",
    A: "Raise the minimum wage to $15.", B: "Let each business decide what to pay.",
    D: [2, -1], R: [-1, 2], I: [1, 1],
    say: [
      ["L", "Finally! Nobody can live on $7.25 an hour.", "My small shop can’t pay that. You just cost people their jobs.", "More money for workers sounds good. Will burgers cost more?", "Workers cheered, but small business owners worry they will have to cut jobs."],
      ["R", "Right. The government shouldn’t set my wages.", "So workers just keep earning too little? I expected better from you.", "Hm, I’m not sure the market will fix this.", "Business owners were relieved, but low-wage workers felt ignored."]
    ] },

  { id: "e3", from: "Parents Group", news: "Parents want phones banned in schools during class",
    A: "Ban phones during class.", B: "Let each teacher decide.",
    D: [2, 0], R: [2, 0], I: [1, 2],
    say: [
      ["N", "Yes! Kids need to pay attention in class.", "What if there’s an emergency and I can’t reach my kid?", "Honestly, I think kids would focus better.", "Most parents liked the ban, though some worried about reaching their kids in an emergency."],
      ["N", "Teachers know their classes best. Good call.", "That’s not a decision, that’s dodging the question.", "Makes sense, every class is different.", "People liked trusting teachers, but some thought you avoided taking a side."]
    ] },

  { id: "e4", from: "City Hall", news: "Should stores stop giving out plastic bags?",
    A: "Ban plastic bags.", B: "Let shoppers choose.",
    D: [2, -1], R: [-1, 2], I: [0, 1],
    say: [
      ["L", "Good. I’m tired of seeing bags in the river.", "Another rule telling me how to shop? Come on.", "I’ll have to remember my own bags… but OK.", "People who care about pollution were happy, but many shoppers found it annoying."],
      ["R", "Thank you. Let people decide for themselves.", "So nothing changes and the bags keep piling up? Disappointing.", "Fine by me, I reuse them anyway.", "Shoppers liked having a choice, but environmental groups called it a missed chance."]
    ] },

  { id: "e5", from: "Gas Station Report", news: "Gas prices jumped 50 cents in one month",
    A: "Pause the gas tax for the summer.", B: "Spend more on buses and trains instead.",
    D: [1, 2], R: [2, -1], I: [2, 0],
    say: [
      ["N", "Every cent helps when you drive to work every day.", "That saves me what, a dollar a tank? Just a stunt.", "Cheaper gas this summer? I’ll take it.", "Drivers liked the quick relief, though some called it a short-term trick."],
      ["L", "Good buses mean I don’t need a car at all.", "I live 20 miles from the nearest bus stop. This does nothing for me.", "Nice idea, but I need help with gas today.", "City people liked better transit, but people in small towns felt it doesn’t help them."]
    ] },

  { id: "e6", from: "City Hall", news: "The city wants to replace street parking with bike lanes",
    A: "Build the bike lanes.", B: "Keep the parking.",
    D: [2, 0], R: [-1, 2], I: [0, 1],
    say: [
      ["L", "Safer streets for kids on bikes. Love it!", "Where am I supposed to park now? I thought you had common sense.", "I might bike to work if it feels safe.", "Cyclists and young families were happy, but drivers and shop owners were angry about losing parking."],
      ["R", "Thank you. My customers need somewhere to park.", "So we wait until someone on a bike gets hit? Really?", "I do drive everywhere, so… fine.", "Drivers and shop owners were relieved, but cyclists said you ignored their safety."]
    ] },

  { id: "e7", from: "Trending", news: "A rival supporter posts a mean meme about you. It gets a million views",
    A: "Laugh it off with a joke.", B: "Post an angry reply.",
    D: [2, 1], R: [2, 1], I: [2, 0],
    say: [
      ["N", "Haha, that reply was better than the meme!", "Too soft. You should hit back harder.", "OK, that was actually funny.", "People liked that you didn’t take yourself too seriously."],
      ["N", "Yes! Don’t let them push you around!", "Wow. Getting this mad over a meme? I thought you were calmer than that.", "That reply felt a bit much.", "Your fans liked the fight, but many voters thought you overreacted."]
    ],
    followup: [null, [
      { p: 0.5, v: -3, who: "Campaign Wire", text: "Your angry reply became a meme of its own. Now it has 5 million views." },
      { p: 0.5, v: 1, who: "Your field team", text: "Your fans loved the comeback, and volunteer sign-ups went up." }
    ]] },

  { id: "e8", from: "Student March", news: "Students march to lower the voting age to 16",
    A: "Let 16-year-olds vote.", B: "Keep the voting age at 18.",
    D: [2, 0], R: [-1, 2], I: [1, 1],
    say: [
      ["L", "We work and pay taxes at 16. We should vote too!", "Sixteen-year-olds can’t even rent a car. Seriously?", "Hm, I’m not sure teens are ready.", "Students loved it, but many older voters think 16 is too young."],
      ["R", "Eighteen is the age you become an adult. Keep it.", "So my future doesn’t matter until I’m 18? Thanks a lot.", "Eighteen sounds right to me.", "Older voters agreed, but young people felt you don’t take them seriously."]
    ] },

  { id: "e9", from: "Business News", news: "A big factory wants to open in town, but it would cut down a forest",
    A: "Welcome the factory and the jobs.", B: "Protect the forest.",
    D: [1, 2], R: [2, -1], I: [2, 0],
    say: [
      ["R", "800 new jobs! My son can finally work here at home.", "We lose the forest forever for one factory? I thought you cared about this town.", "Jobs are good… I’ll miss that forest though.", "Job seekers were excited, but nature lovers felt you sold out the town."],
      ["L", "Thank you. Some things are worth more than money.", "My family needs those jobs more than trees.", "I love that forest, but jobs matter too.", "Nature lovers cheered, but people looking for work were disappointed."]
    ] },

  { id: "e10", from: "Entertainment News", news: "A famous singer offers to endorse you",
    A: "Accept and do a concert together.", B: "Say thank you, but keep the focus on issues.",
    D: [2, 1], R: [2, 1], I: [3, 1],
    say: [
      ["N", "A concert?! I’m registering to vote just for this.", "A concert? Talk about real problems, not celebrities.", "Oh, I love that singer! Interesting…", "Young voters got excited, but some said a concert isn’t a plan."],
      ["N", "Respect. You care about issues, not fame.", "You turned down free attention? Big mistake.", "OK, that’s a serious candidate.", "People respected your focus, but few of them noticed the news."]
    ],
    followup: [[
      { p: 0.5, v: -4, who: "Campaign Wire", text: "Old offensive posts from the singer went viral. Reporters asked why you stood next to them on stage." },
      { p: 0.5, v: 1, who: "Your field team", text: "Thousands of young fans registered to vote after the concert." }
    ], null] },

  { id: "e11", from: "Debate Night", news: "In the debate, you mixed up two numbers. Clips are spreading",
    A: "Admit the mistake right away.", B: "Insist you were right.",
    D: [1, 2], R: [1, 2], I: [1, 1],
    say: [
      ["N", "Everyone makes mistakes. Owning it is rare. Respect.", "Admitting mistakes makes you look weak.", "OK, that’s honest. I like that.", "Most people liked the honesty and quickly moved on."],
      ["N", "Never back down! That’s my candidate.", "You got it wrong and still won’t admit it? I thought you were more honest than that.", "Wait, but the clip clearly shows the mistake…", "Your fans liked the confidence, but undecided voters started to doubt you."]
    ],
    followup: [null, [
      { p: 0.6, v: -4, who: "Fact Check Desk", text: "Fact-checkers showed the clip side by side with the real numbers. Every channel played it." },
      { p: 0.4, v: 0, who: "Campaign Wire", text: "A bigger story took over the news. Nobody asked about the numbers again." }
    ]] },

  { id: "e12", from: "Police Department", news: "Residents ask for more police after a rise in car break-ins",
    A: "Hire more police officers.", B: "Hire more social workers and youth programs.",
    D: [0, 2], R: [2, -1], I: [2, 0],
    say: [
      ["R", "Finally. I want to feel safe on my own street.", "More police won’t fix why people steal. I thought you understood that.", "More patrols would make me feel safer.", "Many residents felt safer, but some said you ignored the causes of crime."],
      ["L", "Give kids something to do and crime goes down. Smart.", "My car got broken into last week and you give me youth programs?", "Hm, that might work in the long run.", "Some liked fixing the causes of crime, but victims of break-ins wanted help now."]
    ] },

  { id: "e13", from: "Health News", news: "A family can’t pay a $10,000 hospital bill",
    A: "The government should cover basic health care for everyone.", B: "Bring prices down with more competition.",
    D: [2, -1], R: [-1, 2], I: [1, 1],
    say: [
      ["L", "Nobody should go broke for getting sick. Thank you!", "Government health care? Higher taxes and long waits. No thanks.", "I’d like not to worry about bills… but how much would it cost?", "People with high medical bills were hopeful, but many worried about taxes."],
      ["R", "More choice and lower prices. That’s how it should work.", "Competition won’t help that family pay its bill today.", "If prices really drop, great.", "Some liked lower prices through competition, but others said it won’t help families fast enough."]
    ] },

  { id: "e14", from: "Student Voices", news: "After a school shooting, students ask leaders to act",
    A: "Background checks for every gun sale.", B: "More security guards in schools.",
    D: [2, -1], R: [-1, 2], I: [2, 0],
    say: [
      ["L", "Checking who buys a gun is the least we can do.", "Criminals don’t do background checks. You’re punishing honest owners.", "Background checks sound like common sense.", "Students and parents liked it, but many gun owners felt targeted."],
      ["R", "Protect the kids where they are. Good.", "Guards at every door? That’s a prison, not a school. I expected more from you.", "I just want my kid to be safe.", "Gun owners agreed, but students said guards alone won’t stop the next shooting."]
    ] },

  { id: "e15", from: "Your Rival’s Ad", news: "Your rival’s new ad says you will raise everyone’s taxes",
    A: "Say you’ll raise taxes only on the very rich.", B: "Promise no new taxes at all.",
    D: [2, 0], R: [-1, 2], I: [1, 1],
    say: [
      ["L", "The rich should pay their fair share. Agreed.", "Today it’s the rich, tomorrow it’s me. I don’t trust that.", "If it’s really only the rich, fine.", "Most people liked that taxes won’t go up for them, but business owners worried."],
      ["R", "No new taxes! Read my lips!", "No taxes? Then who pays for the roads and schools?", "Everyone promises that. Will they keep it?", "People liked the promise, but many doubted you could keep it."]
    ] },

  { id: "e16", from: "Weather Service", news: "A summer heat wave breaks all records",
    A: "Move faster to clean energy.", B: "Keep energy cheap first.",
    D: [2, -1], R: [-1, 2], I: [1, 1],
    say: [
      ["L", "Finally someone takes the heat seriously!", "Clean energy means higher electric bills. I thought you cared about families.", "It is getting hotter every year…", "Young voters cheered, but some families worried about higher energy bills."],
      ["R", "I need my AC and I need it cheap. Thank you.", "Cheap today, but what about my kids’ future?", "Lower bills sound good to me.", "Families liked lower bills, but young voters felt you ignored climate change."]
    ] },

  { id: "e17", from: "Viral Post", news: "A fake post claims you hate dogs. It’s spreading fast",
    A: "Post a photo with your dog and the facts.", B: "Ignore it.",
    D: [2, 0], R: [2, 0], I: [2, 0],
    say: [
      ["N", "Look at that dog! Rumor busted.", "Why even answer a silly rumor?", "Oh, so it was fake. Good to know.", "People liked the quick, friendly answer, and the rumor faded."],
      ["N", "Smart. Don’t feed the trolls.", "Silence? So it’s true?", "Wait, do they really hate dogs?", "Some people started to believe the rumor because you said nothing."]
    ],
    followup: [null, [
      { p: 0.5, v: -2, who: "Campaign Wire", text: "The dog rumor kept spreading. A late-night show made jokes about it." },
      { p: 0.5, v: 0, who: "Campaign Wire", text: "The rumor died out in a few days." }
    ]] },

  { id: "e18", from: "Farm Report", news: "Farmers say they can’t find enough workers for the harvest",
    A: "Give out more work visas.", B: "Train and hire local workers first.",
    D: [2, 0], R: [0, 2], I: [1, 1],
    say: [
      ["L", "Without workers, the crops rot in the field. Good.", "What about Americans who need jobs? I thought you were on our side.", "If the farms need help, OK.", "Farmers were relieved, but some workers felt you ignored them."],
      ["R", "Jobs for our own people first. Agreed.", "The harvest won’t wait for a training program.", "Training locals sounds good, but will it be ready in time?", "Local workers liked it, but farmers worried their crops will rot."]
    ] },

  { id: "e19", from: "College News", news: "College now costs more than a house in some states",
    A: "Make community college free.", B: "Fund more trade schools and job training.",
    D: [2, 1], R: [-1, 2], I: [1, 2],
    say: [
      ["L", "Free community college could change my life.", "Free? Nothing is free. Taxpayers will pay.", "That would help a lot of families.", "Students and parents loved it, but some asked who will pay."],
      ["R", "Not everyone needs college. Electricians make good money!", "So you’re telling kids not to dream of college?", "Good. Trade jobs pay well.", "Many people liked more paths to good jobs, though some wanted help with college costs too."]
    ] },

  { id: "e20", from: "Your Campaign Team", news: "Volunteers ask: knock on doors or post on social media?",
    A: "Knock on doors and talk to people.", B: "Post more on social media.",
    D: [2, 1], R: [2, 1], I: [1, 2],
    say: [
      ["N", "A real person came to my door. That meant a lot.", "Door knocking? It’s not 1990 anymore.", "Someone actually listened to me today.", "People liked meeting someone from your team face to face."],
      ["N", "Your videos are everywhere. Love it!", "Posts are easy to scroll past.", "I saw your post, I guess.", "Your posts reached many people, but most just scrolled past."]
    ] }
];
