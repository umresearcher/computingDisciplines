let articles = [];

const leaningLabels = {
    "-2": "Strongly Left",
    "-1": "Left",
    "0": "Center",
    "1": "Right",
    "2": "Strongly Right"
};

function stanceLabel(score) {

    if (score <= -1.5)
        return "Strongly Opposed";

    if (score <= -0.5)
        return "Slightly Opposed";

    if (score < 0.5)
        return "Balanced";

    if (score < 1.5)
        return "Slightly Supportive";

    return "Strongly Supportive";
}

function levelLabel(score) {

    if (score < 1.5)
        return "Low";

    if (score < 2.5)
        return "Moderate";

    return "High";
}

function diversityLabel(feed) {

    const counts = stanceCounts(feed);

    const values = Object.values(counts);

    const represented =
        values.filter(v => v > 0).length;

    const maxCount =
        Math.max(...values);

    if (represented <= 2)
        return "Low";

    if (represented === 3) {

        if (maxCount >= 5)
            return "Low";

        return "Moderate";
    }

    if (represented === 4) {

        if (maxCount >= 6)
            return "Moderate";

        return "High";
    }

    // all 5 viewpoints represented

    if (maxCount >= 7)
        return "Moderate";

    return "High";
}

/* function diversityLabel(feed) {

    const uniqueStances =
        new Set(
            feed.map(article => article.stance)
        );

    const count =
        uniqueStances.size;

    if (count <= 2)
        return "Low (few viewpoints)";

    if (count <= 4)
        return "Moderate";

    return "High (many viewpoints)";
} */

function stanceCounts(feed) {

    const counts = {
        "strongly oppose": 0,
        "oppose": 0,
        "balanced coverage": 0,
        "support": 0,
        "strongly support": 0
    };

    feed.forEach(article => {
        counts[article.stance]++;
    });

    return counts;
}

function stanceDisplay(stance) {

    switch (stance) {

        case "strongly oppose":
            return "Strongly Opposed";

        case "oppose":
            return "Opposing Viewpoint";

        case "balanced coverage":
            return "Balanced Viewpoint";

        case "support":
            return "Supportive Viewpoint";

        case "strongly support":
            return "Strongly Supportive";
    }
}

function scoreDisplay(score) {

    if (score === 1)
        return "Low";

    return "High";
}

async function loadArticles() {

    const response =
        await fetch("data/articles.json");

    articles =
        await response.json();

    console.log(
        `Loaded ${articles.length} articles`
    );
}

function updateLabels() {

    document.getElementById("leaningValue").textContent =
        leaningLabels[
            document.getElementById("leaning").value
        ];
    /* document.getElementById("leaningValue").textContent =
        document.getElementById("leaning").value;*/

    document.getElementById("engagementValue").textContent =
        document.getElementById("engagementWeight").value;

    document.getElementById("evidenceValue").textContent =
        document.getElementById("evidenceWeight").value;

    document.getElementById("hostilityValue").textContent =
        document.getElementById("hostilityWeight").value;
}

function generateFeed() {

    let userLeaning =
        Number(
            document.getElementById("leaning").value
        );

    let engagementWeight =
        Number(
            document.getElementById("engagementWeight").value
        );

    let evidenceWeight =
        Number(
            document.getElementById("evidenceWeight").value
        );

    let hostilityWeight =
        Number(
            document.getElementById("hostilityWeight").value
        );

    let scoredArticles =
        articles.map(article => {

            let stanceMatch =
                4 -
                Math.abs(
                    userLeaning -
                    article.stance_score
                );

            let score =
                stanceMatch
                + engagementWeight *
                  article.engagement_score
                + evidenceWeight *
                  article.evidence_score
                - hostilityWeight *
                  article.hostility_score;

            return {
                ...article,
                score: score
            };
        });

    scoredArticles.sort(
        (a, b) => b.score - a.score
    );

    let feed =
        scoredArticles.slice(0, 10);

    renderFeed(feed);

    renderStats(feed);
}

function renderFeed(feed) {

    let html = "";

    feed.forEach(article => {

        html += `
            <div class="article">
                <h3>${article.headline}</h3>

                <p>${article.summary}</p>

                <!-- <small>
                    ${article.article_label}
                </small> -->

                <div class="article-tags">

                    <span class="tag">
                        ${stanceDisplay(article.stance)}
                    </span>

                    <span class="tag">
                        ${scoreDisplay(article.engagement_score)}
                        Engagement
                    </span>

                    <span class="tag">
                        ${scoreDisplay(article.hostility_score)}
                        Hostility
                    </span>

                    <span class="tag">
                        ${scoreDisplay(article.evidence_score)}
                        Evidence
                    </span>

                </div>                
            </div>
        `;
    });

    document.getElementById("feed").innerHTML =
        html;
}

function renderStats(feed) {

    const avgStance =
        average(feed, "stance_score");

    const avgEngagement =
        average(feed, "engagement_score");

    const avgHostility =
        average(feed, "hostility_score");

    const avgEvidence =
        average(feed, "evidence_score");

    const counts =
        stanceCounts(feed);

    const viewpointsRepresented =
        Object.values(counts)
            .filter(count => count > 0)
            .length;

    const viewpointCoverage =
        viewpointsRepresented === 5
            ? "All viewpoints represented"
            : viewpointsRepresented >= 3
            ? "Several viewpoints represented"
            : "Limited range of viewpoints";

    const diversity =
        diversityLabel(feed);

    document.getElementById("stats").innerHTML = `

        <div class="stats-container">

            <div class="stats-card">

                <h3>Feed Composition</h3>

                <p><strong>Strongly Opposed:</strong>
                ${counts["strongly oppose"]}</p>

                <p><strong>Opposed:</strong>
                ${counts["oppose"]}</p>

                <p><strong>Balanced:</strong>
                ${counts["balanced coverage"]}</p>

                <p><strong>Supportive:</strong>
                ${counts["support"]}</p>

                <p><strong>Strongly Supportive:</strong>
                ${counts["strongly support"]}</p>

            </div>

            <div class="stats-card">

                <h3>Feed Characteristics</h3>

                <p><strong>Political Balance:</strong>
                ${stanceLabel(avgStance)}</p>

                <p><strong>Viewpoints Represented:</strong>
                ${viewpointsRepresented} of 5</p>

                <!-- <p><strong>Viewpoint Coverage:</strong>
                ${viewpointCoverage}</p> -->

                <!-- <p><strong>Diversity:</strong>
                ${diversity}</p> -->

                <p><strong>Engagement Level:</strong>
                ${levelLabel(avgEngagement)}</p>

                <p><strong>Hostility Level:</strong>
                ${levelLabel(avgHostility)}</p>

                <p><strong>Evidence Level:</strong>
                ${levelLabel(avgEvidence)}</p>

            </div>

        </div>
    `;
}

/* function renderStats(feed) {

    let avgStance =
        average(feed, "stance_score");

    let avgEngagement =
        average(feed, "engagement_score");

    let avgHostility =
        average(feed, "hostility_score");

    let avgEvidence =
        average(feed, "evidence_score");

    document.getElementById("stats").innerHTML = `
        <h3>Feed Summary</h3>

        <p>
            <strong>Political Balance:</strong>
            ${stanceLabel(avgStance)}
        </p>

        <p>
            <strong>Engagement Level:</strong>
            ${levelLabel(avgEngagement)}
        </p>

        <p>
            <strong>Hostility Level:</strong>
            ${levelLabel(avgHostility)}
        </p>

        <p>
            <strong>Evidence Level:</strong>
            ${levelLabel(avgEvidence)}
        </p>

        <p class="summary-text">
        This feed contains mostly
        ${stanceLabel(avgStance).toLowerCase()}
        articles with
        ${levelLabel(avgEvidence).toLowerCase()} use of evidence
        and
        ${levelLabel(avgHostility).toLowerCase()} levels of hostility.
        </p>
    `;

} */

function average(arr, key) {

    let total = 0;

    arr.forEach(
        item => total += item[key]
    );

    return total / arr.length;
}

window.onload = async () => {

    await loadArticles();

    updateLabels();

    document.querySelectorAll(
        "input[type=range]"
    ).forEach(slider => {

        slider.addEventListener(
            "input",
            updateLabels
        );
    });

    document
        .getElementById("generateBtn")
        .addEventListener(
            "click",
            generateFeed
        );
};
