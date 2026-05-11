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
    const titleMap = { 'dashboard': 'Overview', 'analytics': 'Segment Analysis', 'tools': 'Prediction Tools' };
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

        // Fetch and display metrics
        const metricsResponse = await fetch('/metrics');
        const metrics = await metricsResponse.json();
        
        document.getElementById('stat-total').innerText = metrics.total_customers.toLocaleString();
        document.getElementById('stat-recency').innerText = metrics.avg_recency.toFixed(0) + 'd';
        document.getElementById('stat-frequency').innerText = metrics.avg_frequency.toFixed(1);
        document.getElementById('stat-monetary').innerText = '₹' + metrics.avg_monetary.toLocaleString('en-IN', { maximumFractionDigits: 0 });
        
        renderCharts();
        renderCustomerTable();
    } catch (err) {
        console.error('Core Engine Error:', err);
    }
}

function escapeHtml(value) {
    return String(value ?? '').replace(/[&<>"']/g, (char) => ({
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    }[char]));
}

function formatCurrency(value) {
    return '₹' + Number(value || 0).toLocaleString('en-IN', { maximumFractionDigits: 0 });
}

function renderCustomerTable(filter = '') {
    const tbody = document.getElementById('customer-table-body');
    if (!tbody || !dashboardData || !dashboardData.data) return;

    const query = filter.trim().toLowerCase();
    const rows = dashboardData.data
        .filter((customer) => {
            const haystack = `${customer.CustomerID} ${customer.Persona}`.toLowerCase();
            return haystack.includes(query);
        })
        .sort((a, b) => b.Monetary - a.Monetary)
        .slice(0, 100);

    if (!rows.length) {
        tbody.innerHTML = '<tr><td colspan="6">No matching customers found.</td></tr>';
        return;
    }

    tbody.innerHTML = rows.map((customer) => `
        <tr>
            <td>${escapeHtml(customer.CustomerID)}</td>
            <td><span class="persona-pill">${escapeHtml(customer.Persona)}</span></td>
            <td>${Number(customer.Recency).toFixed(0)} days</td>
            <td>${Number(customer.Frequency).toFixed(0)}</td>
            <td>${formatCurrency(customer.Monetary)}</td>
            <td><a class="table-link" href="/customer/${encodeURIComponent(customer.CustomerID)}">View</a></td>
        </tr>
    `).join('');
}

function renderCharts() {
    if (!dashboardData || !dashboardData.data) return;
    
    const isDark = document.body.classList.contains('dark-theme');
    const paperBg = 'rgba(0,0,0,0)';
    const textColor = isDark ? '#f8fafc' : '#0f172a';
    const gridColor = isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.05)';

    const colors = { 0: '#10b981', 1: '#f59e0b', 2: '#3b82f6', 3: '#ef4444' };
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
            // Store CustomerID in customdata for click events
            customdata: cData.map(item => item.CustomerID),
            hoverinfo: 'text',
            text: cData.map(item => `ID: ${item.CustomerID}<br>Persona: ${item.Persona}`),
            mode: 'markers',
            marker: { size: 4, color: colors[c] || '#64748b', opacity: 0.8 },
            type: 'scatter3d',
            name: personaMap[c] || `Segment ${c}`
        });
    });

    const layout3d = {
        scene: {
            xaxis: { title: 'Recency (Days)', color: textColor, gridcolor: gridColor },
            yaxis: { title: 'Frequency', color: textColor, gridcolor: gridColor },
            zaxis: { title: 'Monetary (₹)', color: textColor, gridcolor: gridColor },
            bgcolor: paperBg
        },
        margin: { l: 0, r: 0, b: 0, t: 0 },
        paper_bgcolor: paperBg,
        font: { family: 'Outfit, sans-serif', color: textColor },
        legend: { orientation: 'h', y: 0.95, font: { size: 12 } }
    };

    Plotly.newPlot('chart-3d', traces, layout3d, { responsive: true, displayModeBar: false });

    // Add click event listener to the 3D chart
    const chart3d = document.getElementById('chart-3d');
    chart3d.on('plotly_click', function(data) {
        if (data.points.length > 0) {
            const point = data.points[0];
            const customerId = point.customdata;
            if (customerId) {
                // Redirect to the customer profile page
                window.location.href = `/customer/${customerId}`;
            }
        }
    });

    // Analytics Tabs
    loadAnalyticsCharts(isDark, textColor, paperBg, colors, personaMap);
}

async function loadAnalyticsCharts(isDark, textColor, paperBg, clusterColors, personaMap) {
    if (!dashboardData || !dashboardData.data) return;

    const segmentData = dashboardData.data;
    const metrics = Object.keys(personaMap).map(clusterId => {
        const clusterData = segmentData.filter(d => d.Cluster == clusterId);
        return {
            Cluster: parseInt(clusterId),
            Count: clusterData.length,
            Monetary: clusterData.reduce((sum, d) => sum + d.Monetary, 0) / clusterData.length || 0
        };
    });

    const cColors = metrics.map(m => clusterColors[m.Cluster] || '#334155');

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
    }, { responsive: true, displayModeBar: false });

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
    }, { responsive: true, displayModeBar: false });
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
            "Champions": "Top-tier customers. Offer exclusive rewards and early access.",
            "Potential Loyalists": "High potential. Nurture with loyalty programs and personalized offers.",
            "At-Risk Customers": "Losing engagement. Re-engage with targeted promotions and feedback surveys.",
            "Lost Customers": "Inactive. Attempt to win back with special offers or acknowledge their departure."
        };

        resPersona.innerText = result.persona;
        resAdvice.innerText = adviceMap[result.persona] || "This customer segment requires further analysis to determine the best marketing strategy.";
        resDiv.style.display = 'block';
    } catch (err) { alert(err.message); }
});

// Churn Predictor Logic
document.getElementById('churn-predictor-form')?.addEventListener('submit', async (e) => {
    e.preventDefault();
    const resDiv = document.getElementById('churn-prediction-result');
    const resPrediction = document.getElementById('churn-res-prediction');
    const resAdvice = document.getElementById('churn-res-advice');

    const data = {
        recency: document.getElementById('churn-recency').value,
        frequency: document.getElementById('churn-frequency').value,
        monetary: document.getElementById('churn-monetary').value
    };

    try {
        const resp = await fetch('/predict_churn', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        const result = await resp.json();

        if (result.error) throw new Error(result.error);

        const probability = (result.churn_probability * 100).toFixed(1);
        if (result.churn_prediction === 1) {
            resPrediction.innerText = `High Risk (${probability}%)`;
            resAdvice.innerText = "This customer is at high risk of churning. Launch a win-back campaign immediately.";
            resDiv.className = "insight-card glass cluster-lost";
        } else {
            resPrediction.innerText = `Low Risk (${probability}%)`;
            resAdvice.innerText = "This customer is likely to stay. Nurture them with loyalty programs.";
            resDiv.className = "insight-card glass cluster-champions";
        }
        
        resDiv.style.display = 'block';
    } catch (err) { 
        alert(err.message); 
    }
});

function setUploadStatus(message, type = 'success') {
    document.querySelectorAll('.upload-status').forEach((status) => {
        status.innerText = message;
        status.className = `upload-status status-msg status-${type}`;
        status.hidden = false;
    });
}

async function uploadCsv(file) {
    if (!file) return;
    if (!file.name.toLowerCase().endsWith('.csv')) {
        setUploadStatus('Please choose a CSV file.', 'error');
        return;
    }

    const zeroState = document.getElementById('zero-state');
    const dashContainer = document.getElementById('dashboard-container');

    const formData = new FormData();
    formData.append('file', file);

    setUploadStatus('Processing data...', 'success');

    try {
        const resp = await fetch('/upload', { method: 'POST', body: formData });
        const result = await resp.json();

        if (result.error) throw new Error(result.error);

        setUploadStatus(`Processed ${result.customers || 'your'} customers. Loading dashboard...`, 'success');
        zeroState.style.display = 'none';
        dashContainer.style.display = 'block';
        openTab(null, 'dashboard');
        setTimeout(() => {
            loadDashboard();
        }, 500);
    } catch (err) {
        setUploadStatus("Upload failed: " + err.message, 'error');
    }
}

// Upload Logic
document.getElementById('csv-upload')?.addEventListener('change', async (e) => {
    await uploadCsv(e.target.files[0]);
    e.target.value = '';
});

const dropZone = document.getElementById('drop-zone');
dropZone?.addEventListener('click', () => document.getElementById('csv-upload').click());
dropZone?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        document.getElementById('csv-upload').click();
    }
});
['dragenter', 'dragover'].forEach((eventName) => {
    dropZone?.addEventListener(eventName, (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });
});
['dragleave', 'drop'].forEach((eventName) => {
    dropZone?.addEventListener(eventName, (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
    });
});
dropZone?.addEventListener('drop', (e) => uploadCsv(e.dataTransfer.files[0]));

document.getElementById('customer-search')?.addEventListener('input', (e) => {
    renderCustomerTable(e.target.value);
});

// Theme
document.getElementById('theme-toggle')?.addEventListener('change', (e) => {
    document.body.className = e.target.checked ? 'dark-theme' : 'light-theme';
    renderCharts();
});

document.addEventListener('DOMContentLoaded', loadDashboard);
