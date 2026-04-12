let dashboardData = null;

function openTab(evt, tabName) {
    const contents = document.getElementsByClassName("tab-content");
    for (let content of contents) content.classList.remove("active");

    const buttons = document.getElementsByClassName("nav-btn");
    for (let btn of buttons) btn.classList.remove("active");

    const target = document.getElementById(tabName);
    if (target) target.classList.add("active");
    
    // Update Nav
    const activeBtn = evt ? evt.currentTarget : document.getElementById('nav-' + tabName);
    if (activeBtn) activeBtn.classList.add("active");

    // Title mapping
    const titleMap = { 'dashboard': 'Overview', 'analytics': 'Segment Analysis', 'tools': 'Insight Tools' };
    document.getElementById('view-title').innerText = titleMap[tabName] || tabName;

    // Trigger Resize for Plotly
    window.dispatchEvent(new Event('resize'));
}

async function loadDashboard() {
    try {
        const response = await fetch('/segment');
        dashboardData = await response.json();

        const zeroState = document.getElementById('zero-state');
        const dashContainer = document.getElementById('dashboard-container');

        if (dashboardData.error === "empty_state") {
            zeroState.style.display = 'flex';
            dashContainer.style.display = 'none';
            return;
        }

        // Show dashboard
        zeroState.style.display = 'none';
        dashContainer.style.display = 'block';

        // Stats
        document.getElementById('stat-total').innerText = dashboardData.stats.total_customers.toLocaleString();
        document.getElementById('stat-recency').innerText = dashboardData.stats.avg_recency.toFixed(0) + 'd';
        document.getElementById('stat-frequency').innerText = dashboardData.stats.avg_frequency.toFixed(1);
        document.getElementById('stat-monetary').innerText = '₹' + dashboardData.stats.avg_monetary.toLocaleString('en-IN', { maximumFractionDigits: 0 });

        renderCharts();
    } catch (err) {
        console.error('Core Engine Error:', err);
    }
}

function renderCharts() {
    if (!dashboardData || !dashboardData.data) return;
    
    const isDark = document.body.classList.contains('dark-theme');
    const paperBg = 'rgba(0,0,0,0)';
    const textColor = isDark ? '#f8fafc' : '#0f172a';
    const gridColor = isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.05)';

    const colors = { 0: '#f59e0b', 1: '#10b981', 2: '#ef4444', 3: '#6366f1' };
    const personaMap = dashboardData.persona_map;

    // 3D Chart
    const traces = [];
    const clusters = [...new Set(dashboardData.data.map(item => item.Cluster))];

    clusters.forEach(c => {
        const cData = dashboardData.data.filter(item => item.Cluster === c);
        traces.push({
            x: cData.map(item => item.Recency),
            y: cData.map(item => item.Frequency),
            z: cData.map(item => item.Monetary),
            mode: 'markers',
            marker: { size: 4, color: colors[c], opacity: 0.8, line: { color: colors[c], width: 1 } },
            type: 'scatter3d',
            name: personaMap[c]
        });
    });

    const layout3d = {
        scene: {
            xaxis: { title: 'Recency (Days)', color: textColor, gridcolor: gridColor },
            yaxis: { title: 'Frequency (Log)', color: textColor, gridcolor: gridColor },
            zaxis: { title: 'Monetary (Log)', color: textColor, gridcolor: gridColor },
            bgcolor: paperBg
        },
        margin: { l: 0, r: 0, b: 0, t: 0 },
        paper_bgcolor: paperBg,
        font: { family: 'Outfit, sans-serif', color: textColor },
        legend: { orientation: 'h', y: 0.95, font: { size: 12 } }
    };

    Plotly.newPlot('chart-3d', traces, layout3d);

    // Analytics Tabs
    loadAnalyticsCharts(isDark, textColor, paperBg, colors, personaMap);
}

async function loadAnalyticsCharts(isDark, textColor, paperBg, clusterColors, personaMap) {
    try {
        const resp = await fetch('/metrics');
        const metrics = await resp.json();
        
        const cColors = [clusterColors[0], clusterColors[1], clusterColors[2], clusterColors[3]];

        // Pie
        const pieData = [{
            values: metrics.map(m => m.Count),
            labels: metrics.map(m => personaMap[m.Cluster]),
            type: 'pie',
            marker: { colors: cColors, line: { color: isDark ? '#020617' : '#fff', width: 2 } },
            hole: 0.5,
            textinfo: 'percent+label'
        }];
        Plotly.newPlot('chart-pie', pieData, { 
            paper_bgcolor: paperBg, 
            font: { family: 'Outfit', color: textColor },
            margin: { t: 40, b: 0, l: 0, r: 0 },
            showlegend: false
        });

        // Bar
        const barData = [{
            x: metrics.map(m => personaMap[m.Cluster]),
            y: metrics.map(m => m.Monetary),
            type: 'bar',
            marker: { color: cColors, line: { width: 0 } },
            text: metrics.map(m => '₹' + m.Monetary.toFixed(0)),
            textposition: 'auto'
        }];
        Plotly.newPlot('chart-bar', barData, { 
            paper_bgcolor: paperBg, 
            plot_bgcolor: paperBg,
            font: { family: 'Outfit', color: textColor },
            yaxis: { title: 'Avg Monetary (₹)', gridcolor: isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)' },
            xaxis: { gridcolor: 'transparent' },
            margin: { t: 40, b: 40, l: 60, r: 20 }
        });

    } catch (err) { console.error('Analytics Error:', err); }
}

// Predictor Logic
document.getElementById('predictor-form')?.addEventListener('submit', async (e) => {
    e.preventDefault();
    const resDiv = document.getElementById('prediction-result');
    const resPersona = document.getElementById('res-persona');
    const resAdvice = document.getElementById('res-advice');

    const data = {
        recency: document.getElementById('pred-recency').value,
        frequency: document.getElementById('pred-frequency').value,
        monetary: document.getElementById('pred-monetary').value
    };

    try {
        const resp = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        const result = await resp.json();

        if (result.error) throw new Error(result.error);

        const adviceMap = {
            "Champions": "Your core revenue drivers. Priority rewards and early access required.",
            "Big Spenders": "High-value occasional buyers. Target with luxury high-ticket bundles.",
            "About to Sleep": "Dormant loyalists. Automated re-engagement codes required.",
            "Potential Loafers": "Trial users. Use volume discounts to increase initial size."
        };

        resPersona.innerText = result.persona;
        resAdvice.innerText = adviceMap[result.persona] || "Analyze behavior for marketing strategy.";
        resDiv.style.display = 'block';
    } catch (err) { alert(err.message); }
});

// Upload Logic
document.getElementById('csv-upload')?.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const status = document.getElementById('upload-status');
    const zeroState = document.getElementById('zero-state');
    const dashContainer = document.getElementById('dashboard-container');

    // If we're in zero-state, show the tools tab so the user sees progress
    if (zeroState.style.display !== 'none') {
        zeroState.style.display = 'none';
        dashContainer.style.display = 'block';
        openTab(null, 'tools');
    }

    const formData = new FormData();
    formData.append('file', file);

    status.innerText = "PRISM is priming the behavioral model...";
    status.className = "status-msg status-success";
    status.style.display = 'block';

    try {
        const resp = await fetch('/upload', { method: 'POST', body: formData });
        const result = await resp.json();

        if (result.error) throw new Error(result.error);

        status.innerText = "Transformation complete. Entering PRISM...";
        setTimeout(() => {
            loadDashboard();
            openTab(null, 'dashboard');
        }, 1500);
    } catch (err) {
        status.innerText = "Upload failed: " + err.message;
        status.className = "status-msg status-error";
    }
});

// Theme
document.getElementById('theme-toggle')?.addEventListener('change', (e) => {
    document.body.className = e.target.checked ? 'dark-theme' : 'light-theme';
    renderCharts();
});

document.addEventListener('DOMContentLoaded', loadDashboard);
