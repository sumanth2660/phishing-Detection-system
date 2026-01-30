// Background Service Worker

const API_URL = "http://localhost:8000/api/v1/predict/url";
const WS_URL = "ws://localhost:8000/api/v1/ws/scan";

let ws = null;
let isConnected = false;

// Initialize WebSocket connection
function connectWebSocket() {
    try {
        ws = new WebSocket(WS_URL);

        ws.onopen = () => {
            console.log("NeuroPhish: Connected to backend");
            isConnected = true;
            broadcastStatus();
        };

        ws.onclose = () => {
            console.log("NeuroPhish: Disconnected");
            isConnected = false;
            broadcastStatus();
            // Reconnect after 5 seconds
            setTimeout(connectWebSocket, 5000);
        };

        ws.onerror = (error) => {
            console.error("NeuroPhish: WebSocket error", error);
            isConnected = false;
        };

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            handleServerMessage(data);
        };

    } catch (e) {
        console.error("Connection failed:", e);
        setTimeout(connectWebSocket, 5000);
    }
}

// Helper to safely send messages to tabs, injecting content script if needed
async function sendMessageToTab(tabId, message) {
    if (!tabId) return;

    try {
        // Try sending message first
        await chrome.tabs.sendMessage(tabId, message);
    } catch (error) {
        console.warn("Content script not ready, attempting injection...", error.message);

        try {
            // Check if we can inject (avoid restricted URLs)
            const tab = await chrome.tabs.get(tabId);
            if (!tab || !tab.url) return;

            if (tab.url.startsWith("chrome://") || tab.url.startsWith("edge://") || tab.url.startsWith("about:")) {
                console.log("Cannot inject into restricted URL:", tab.url);
                return;
            }

            // Inject content script dynamically
            await chrome.scripting.executeScript({
                target: { tabId: tabId },
                files: ['content.js']
            });

            // Retry sending message after injection
            // Give it a tiny delay to ensure script initializes
            await new Promise(resolve => setTimeout(resolve, 100));
            await chrome.tabs.sendMessage(tabId, message);

        } catch (retryError) {
            console.error("Failed to inject or retry message:", retryError.message);
        }
    }
}

function handleServerMessage(data) {
    if (data.type === "scan_result") {
        // Find active tab to show result
        chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
            if (tabs[0]) {
                sendMessageToTab(tabs[0].id, {
                    action: "show_result_toast",
                    result: data
                });
            }
        });
    }
}

function broadcastStatus() {
    chrome.runtime.sendMessage({
        action: "status_update",
        connected: isConnected
    }).catch(() => { }); // Ignore error if no popup open
}

// Connect on startup
connectWebSocket();

// Listen for messages
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "analyze_url") {
        // Use WebSocket if available, else fallback to HTTP
        if (isConnected && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ url: request.url }));
            sendResponse({ status: "queued" });
        } else {
            analyzeUrlHTTP(request.url)
                .then(result => sendResponse(result))
                .catch(error => sendResponse({ error: error.message }));
            return true; // Async response
        }
    } else if (request.action === "get_status") {
        sendResponse({ connected: isConnected });
    }
});

async function analyzeUrlHTTP(url) {
    try {
        const response = await fetch(API_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ url: url, context: "browser_extension" })
        });
        if (!response.ok) throw new Error(`API Error: ${response.status}`);
        return await response.json();
    } catch (error) {
        throw error;
    }
}

// Context menu
chrome.runtime.onInstalled.addListener(() => {
    if (chrome.contextMenus) {
        chrome.contextMenus.create({
            id: "scan_link",
            title: "Scan with NeuroPhish",
            contexts: ["all"]
        });
    }
});

if (chrome.contextMenus) {
    chrome.contextMenus.onClicked.addListener((info, tab) => {
        if (info.menuItemId === "scan_link") {
            const targetUrl = info.linkUrl || info.pageUrl;
            if (!targetUrl) return;

            sendMessageToTab(tab.id, { action: "show_scanning_toast" });

            if (isConnected && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ url: targetUrl }));
            } else {
                analyzeUrlHTTP(targetUrl)
                    .then(result => {
                        sendMessageToTab(tab.id, {
                            action: "show_result_toast",
                            result: result
                        });
                    })
                    .catch(error => {
                        sendMessageToTab(tab.id, {
                            action: "show_error_toast",
                            error: error.message
                        });
                    });
            }
        }
    });
}
