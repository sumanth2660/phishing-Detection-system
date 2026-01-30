document.addEventListener('DOMContentLoaded', function () {
    const scanBtn = document.getElementById('scan-btn');
    const statusIndicator = document.getElementById('connection-status');
    const alertsList = document.getElementById('alerts-list');

    // Check connection status
    chrome.runtime.sendMessage({ action: "get_status" }, function (response) {
        updateStatus(response && response.connected);
    });

    // Listen for status updates
    chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
        if (request.action === "status_update") {
            updateStatus(request.connected);
        }
    });

    scanBtn.addEventListener('click', function () {
        chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
            if (tabs[0]) {
                chrome.tabs.sendMessage(tabs[0].id, { action: "scan_page" }, (response) => {
                    if (chrome.runtime.lastError) {
                        // Content script not ready? Inject it now!
                        chrome.scripting.executeScript({
                            target: { tabId: tabs[0].id },
                            files: ['content.js']
                        }, () => {
                            // Retry sending the message after injection
                            setTimeout(() => {
                                chrome.tabs.sendMessage(tabs[0].id, { action: "scan_page" });
                                window.close();
                            }, 100);
                        });
                    } else {
                        window.close();
                    }
                });
            }
        });
    });

    function updateStatus(connected) {
        if (connected) {
            statusIndicator.className = 'status-indicator connected';
            statusIndicator.innerHTML = '<div class="dot"></div><span>Connected</span>';
            scanBtn.disabled = false;
        } else {
            statusIndicator.className = 'status-indicator disconnected';
            statusIndicator.innerHTML = '<div class="dot"></div><span>Disconnected</span>';
            scanBtn.disabled = true;
        }
    }
});
