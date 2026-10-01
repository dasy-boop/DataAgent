const input = document.getElementById("keyword");
const button = document.getElementById("analyzeButton");
const message = document.getElementById("message");
const tableBody = document.getElementById("skillTableBody");

button.addEventListener("click", async () => {
    const keyword = input.value.trim();

    if (!keyword) {
        message.textContent = "Please enter a keyword";
        return;
    }

    message.textContent = "Loading...";
    tableBody.innerHTML = "";

    try {
        const url =
            "/analysis/skills?keyword=" +
            encodeURIComponent(keyword) +
            "&top_n=10";

        const response = await fetch(url);
        const data = await response.json();

        if (!response.ok) {
            throw new Error("Request failed");
        }

        message.textContent =
            "Matched: " + data.matched_records +
            ", valid skills: " + data.valid_skill_records;

        for (const item of data.skills) {
            const values = Object.values(item);
            const row = document.createElement("tr");

            row.innerHTML =
                "<td>" + values[0] + "</td>" +
                "<td>" + values[1] + "</td>" +
                "<td>" + values[2] + "%</td>";

            tableBody.appendChild(row);
        }
    } catch (error) {
        message.textContent = "Analysis failed";
        console.error(error);
    }
});
