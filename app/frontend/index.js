const API_URL = "http://127.0.0.1:8000/search?query=";

function handleKeyPress(event) {
  if (event.key === "Enter") {
    performSearch();
  }
}
function clearInput() {
  document.getElementById("search-input").value = "";
}
async function performSearch() {
  const query = document.getElementById("search-input").value.trim();
  if (!query) return;

  const resultsContainer = document.getElementById("results-container");
  const loadingIndicator = document.getElementById("loading");

  // Reset UI
  resultsContainer.innerHTML = "";
  loadingIndicator.classList.remove("hidden");

  try {
    // Fetch the JSON from FastAPI backend
    const response = await fetch(API_URL + encodeURIComponent(query));
    const data = await response.json();

    loadingIndicator.classList.add("hidden");
    renderResults(data.results);
  } catch (error) {
    loadingIndicator.classList.add("hidden");
    resultsContainer.innerHTML = `<p class="text-red-500 text-center">Error connecting to the search engine. Is FastAPI running?</p>`;
    console.error("Search Error:", error);
  }
}

function renderResults(results) {
  const container = document.getElementById("results-container");
  if (results.length === 0) {
    container.innerHTML = `<p class="text-gray-500 text-center">No results found.</p>`;
    return;
  }

  results.forEach((result, index) => {
    // Format the score to 4 decimal places
    const score = parseFloat(result.score).toFixed(4);

    // Create the HTML card for each paper
    const card = `
            <div class="bg-white p-6 rounded-lg shadow-sm border border-gray-200 hover:shadow-md transition-shadow">
                <div class="flex justify-between items-start mb-2">
                    <h2 class="text-xl font-semibold text-blue-800 leading-tight">
                        <a href="https://arxiv.org/abs/${result.doc_id}" target="_blank" class="hover:underline">
                            ${result.title}
                        </a>
                    </h2>
                    <span class="text-xs font-mono bg-gray-100 text-gray-600 px-2 py-1 rounded">Score: ${score}</span>
                </div>
                <div class="text-sm text-green-700 mb-3 font-mono">arxiv.org/abs/${result.doc_id}</div>
                <p class="text-gray-600 text-sm leading-relaxed line-clamp-3">
                    ${result.abstract}
                </p>
            </div>
        `;
    container.insertAdjacentHTML("beforeend", card);
  });
}
