// Theme handling
function initTheme() {
    const theme = localStorage.getItem('theme') || 'light';
    document.documentElement.setAttribute('data-theme', theme);
    updateThemeIcon(theme);
}

function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute('data-theme');
    const newTheme = currentTheme === 'light' ? 'dark' : 'light';
    
    document.documentElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
    updateThemeIcon(newTheme);
    
    resizeCharts();
}

function updateThemeIcon(theme) {
    const icon = document.querySelector('#theme-toggle i');
    icon.className = theme === 'light' ? 'fas fa-moon' : 'fas fa-sun';
}

// Search functionality
function initializeSearch() {
    const searchInput = document.getElementById('stock-search');
    const suggestionsContainer = document.getElementById('search-suggestions');
    let selectedIndex = -1;

    searchInput.addEventListener('input', async function(e) {
        const query = e.target.value.trim();
        if (!query) {
            suggestionsContainer.classList.remove('active');
            return;
        }

        try {
            const response = await fetch('/search_ticker', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query: query })
            });
            
            const suggestions = await response.json();
            displaySuggestions(suggestions);
            selectedIndex = -1;
        } catch (error) {
            console.error('Error fetching suggestions:', error);
        }
    });

    searchInput.addEventListener('keydown', function(e) {
        const suggestions = document.querySelectorAll('.suggestion-item');
        
        switch(e.key) {
            case 'ArrowDown':
                e.preventDefault();
                selectedIndex = Math.min(selectedIndex + 1, suggestions.length - 1);
                updateSelection(suggestions);
                break;
            case 'ArrowUp':
                e.preventDefault();
                selectedIndex = Math.max(selectedIndex - 1, -1);
                updateSelection(suggestions);
                break;
            case 'Enter':
                e.preventDefault();
                if (selectedIndex >= 0 && suggestions[selectedIndex]) {
                    selectSuggestion(suggestions[selectedIndex].dataset.symbol);
                } else if (suggestions.length > 0) {
                    selectSuggestion(suggestions[0].dataset.symbol);
                }
                break;
            case 'Escape':
                suggestionsContainer.classList.remove('active');
                selectedIndex = -1;
                break;
        }
    });

    document.addEventListener('click', function(e) {
        if (!searchInput.contains(e.target) && !suggestionsContainer.contains(e.target)) {
            suggestionsContainer.classList.remove('active');
        }
    });

    function displaySuggestions(suggestions) {
        if (suggestions.length === 0) {
            suggestionsContainer.classList.remove('active');
            return;
        }

        suggestionsContainer.innerHTML = suggestions
            .map(item => `
                <div class="suggestion-item" data-symbol="${item.symbol}">
                    <strong>${item.symbol}</strong> - ${item.name}
                </div>
            `)
            .join('');

        suggestionsContainer.classList.add('active');

        const suggestionItems = document.querySelectorAll('.suggestion-item');
        suggestionItems.forEach(item => {
            item.addEventListener('click', () => selectSuggestion(item.dataset.symbol));
        });
    }

    function updateSelection(suggestions) {
        suggestions.forEach((suggestion, index) => {
            suggestion.classList.toggle('selected', index === selectedIndex);
            if (index === selectedIndex) {
                suggestion.scrollIntoView({ block: 'nearest' });
            }
        });
    }

    function selectSuggestion(symbol) {
        searchInput.value = symbol;
        suggestionsContainer.classList.remove('active');
        getPrediction();
    }
}

async function validateTicker(symbol) {
    try {
        const response = await fetch('/validate_ticker', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ symbol: symbol })
        });
        const data = await response.json();
        return data.valid;
    } catch (error) {
        console.error('Error validating ticker:', error);
        return false;
    }
}

async function getPrediction() {
    const searchInput = document.getElementById('stock-search');
    const symbol = searchInput.value.trim().toUpperCase();
    const resultDiv = document.getElementById('result');
    const errorDiv = document.getElementById('error-message');
    const graphsContainer = document.getElementById('graphs-container');
    const errorText = errorDiv.querySelector('.error-text');
    
    if (!symbol) {
        errorText.textContent = 'Please enter a stock symbol';
        errorDiv.classList.add('visible');
        resultDiv.classList.add('hidden');
        graphsContainer.classList.add('hidden');
        return;
    }
    
    const isValid = await validateTicker(symbol);
    if (!isValid) {
        errorText.textContent = 'Invalid stock symbol. Please enter a valid symbol.';
        errorDiv.classList.add('visible');
        resultDiv.classList.add('hidden');
        graphsContainer.classList.add('hidden');
        return;
    }
    
    errorDiv.classList.remove('visible');
    
    const button = document.querySelector('.primary-button');
    button.disabled = true;
    button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: `symbol=${symbol}`
        });
        
        const data = await response.json();
        
        if (data.error) {
            errorText.textContent = data.error;
            errorDiv.classList.add('visible');
            resultDiv.classList.add('hidden');
            graphsContainer.classList.add('hidden');
        } else {
            document.getElementById('result-symbol').textContent = data.symbol;
            document.getElementById('result-prediction').textContent = data.prediction;
            document.getElementById('buy-probability').textContent = data.buy_probability;
            document.getElementById('sell-probability').textContent = data.sell_probability;
            
            const isDarkMode = document.documentElement.getAttribute('data-theme') === 'dark';
            
            const config = {
                responsive: true,
                displayModeBar: true,
                modeBarButtonsToAdd: [
                    'drawline',
                    'drawopenpath',
                    'drawclosedpath',
                    'drawcircle',
                    'drawrect',
                    'eraseshape'
                ],
                modeBarButtonsToRemove: ['lasso2d'],
                displaylogo: false
            };

            const commonLayoutUpdates = {
                margin: { l: 50, r: 20, t: 30, b: 30 },
                autosize: true,
                showlegend: true,
                legend: {
                    orientation: 'h',
                    yanchor: 'bottom',
                    y: 1.02,
                    xanchor: 'right',
                    x: 1
                },
                paper_bgcolor: isDarkMode ? '#1E1E1E' : '#FFFFFF',
                plot_bgcolor: isDarkMode ? '#1E1E1E' : '#FFFFFF',
                font: {
                    color: isDarkMode ? '#FFFFFF' : '#333333'
                }
            };

            Object.entries(data.graphs).forEach(([chartName, chartData]) => {
                const layout = {
                    ...chartData.layout,
                    ...commonLayoutUpdates
                };
                Plotly.newPlot(`${chartName}-chart`, chartData.data, layout, config);
            });
            
            resultDiv.classList.remove('hidden');
            graphsContainer.classList.remove('hidden');
            errorDiv.classList.remove('visible');
            
            setTimeout(resizeCharts, 100);
        }
    } catch (error) {
        errorText.textContent = 'An error occurred while making the prediction.';
        errorDiv.classList.add('visible');
        resultDiv.classList.add('hidden');
        graphsContainer.classList.add('hidden');
    } finally {
        button.disabled = false;
        button.innerHTML = '<i class="fas fa-search"></i> Analyze Stock';
    }
}

function resizeCharts() {
    const charts = ['price-chart', 'volume-chart', 'rsi-chart'];
    charts.forEach(chartId => {
        const chart = document.getElementById(chartId);
        if (chart && chart.data) {
            Plotly.Plots.resize(chart);
        }
    });
}

function toggleDrawingMode(chartId) {
    const chart = document.getElementById(chartId);
    if (chart && chart.layout) {
        const newMode = chart.layout.dragmode === 'drawline' ? 'zoom' : 'drawline';
        Plotly.relayout(chartId, {dragmode: newMode});
        
        const button = document.querySelector(`button[onclick="toggleDrawingMode('${chartId}')"]`);
        if (button) {
            button.classList.toggle('active');
        }
    }
}

function clearDrawings(chartId) {
    const chart = document.getElementById(chartId);
    if (chart && chart.layout) {
        Plotly.relayout(chartId, {shapes: []});
    }
}

// Event Listeners
window.addEventListener('resize', resizeCharts);

document.addEventListener('DOMContentLoaded', function() {
    initTheme();
    initializeSearch();
    
    document.getElementById('theme-toggle').addEventListener('click', toggleTheme);
});