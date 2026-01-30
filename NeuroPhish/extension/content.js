// Content Script

// Prevent double injection
if (window.neuroPhishInitialized) {
  console.log("NeuroPhish already initialized");
} else {
  window.neuroPhishInitialized = true;
  console.log("NeuroPhish content script loaded");

  try {
    initializeToastStyles();
    setupMessageListener();
  } catch (e) {
    console.error("NeuroPhish initialization failed:", e);
  }
}

function initializeToastStyles() {
  const css = `
    .neurophish-toast {
      position: fixed;
      bottom: 20px;
      right: 20px;
      background: white;
      border-radius: 8px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.15);
      padding: 16px;
      z-index: 2147483647; /* Max z-index */
      font-family: system-ui, -apple-system, sans-serif;
      max-width: 350px;
      border-left: 4px solid #3b82f6;
      animation: slideIn 0.3s ease-out;
      pointer-events: auto;
    }
    
    .neurophish-toast.risk-critical { border-left-color: #ef4444; }
    .neurophish-toast.risk-high { border-left-color: #f97316; }
    .neurophish-toast.risk-medium { border-left-color: #eab308; }
    .neurophish-toast.risk-low { border-left-color: #22c55e; }
    
    .neurophish-title {
      font-weight: 600;
      margin-bottom: 4px;
      color: #1f2937;
      font-size: 16px;
    }
    
    .neurophish-message {
      font-size: 14px;
      color: #4b5563;
    }
    
    @keyframes slideIn {
      from { transform: translateX(100%); opacity: 0; }
      to { transform: translateX(0); opacity: 1; }
    }
  `;

  const style = document.createElement('style');
  style.textContent = css;

  // Safe injection
  const target = document.head || document.documentElement;
  if (target) {
    target.appendChild(style);
  } else {
    console.warn("NeuroPhish: Could not find head or documentElement to inject styles");
  }
}

function setupMessageListener() {
  // Listen for messages from background script
  chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    // ... existing listener logic ...
    handleMessage(request);
    sendResponse({ status: "received" });
    return true; // Keep channel open
  });
}

function handleMessage(request) {
  if (request.action === "show_scanning_toast") {
    showToast("Scanning Link...", "Analyzing URL for threats...", "info");
  } else if (request.action === "show_result_toast") {
    const result = request.result;
    const riskLevel = result.risk_level;
    const score = Math.round(result.probability * 100);

    let title = `Risk Score: ${score}%`;
    let message = "This link appears safe.";
    let type = "low";

    if (riskLevel === "critical" || riskLevel === "high") {
      title = `⚠️ High Risk Detected (${score}%)`;
      message = "This link is dangerous! Do not visit.";
      type = "critical";
    } else if (riskLevel === "medium") {
      title = `⚠️ Caution Advised (${score}%)`;
      message = "This link shows suspicious traits.";
      type = "medium";
    }

    showToast(title, message, type);
  } else if (request.action === "show_error_toast") {
    showToast("Analysis Failed", request.error, "error");
  }
}

function showToast(title, message, type) {
  // Remove existing toast
  const existing = document.querySelector('.neurophish-toast');
  if (existing) existing.remove();

  const toast = document.createElement('div');
  toast.className = `neurophish-toast risk-${type}`;

  toast.innerHTML = `
    <div class="neurophish-title">${title}</div>
    <div class="neurophish-message">${message}</div>
  `;

  document.body.appendChild(toast);

  // Auto remove after 5 seconds
  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateX(100%)';
    toast.style.transition = 'all 0.3s ease-in';
    setTimeout(() => toast.remove(), 300);
  }, 5000);
}
