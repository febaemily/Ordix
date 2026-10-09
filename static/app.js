/**
 * ORDIX — Orders Simplified
 * Minimalist AI Chatbot Application
 * Pure Vanilla JavaScript • Fast • Grounded in orders.csv
 */

document.addEventListener("DOMContentLoaded", () => {
  // ==========================================================================
  // DOM References
  // ==========================================================================
  const chatForm = document.getElementById("chatForm");
  const chatInput = document.getElementById("chatInput");
  const sendBtn = document.getElementById("sendBtn");
  const chatContainer = document.getElementById("chatContainer");
  const messagesList = document.getElementById("messagesList");
  const welcomeScreen = document.getElementById("welcomeScreen");
  const loadingIndicator = document.getElementById("loadingIndicator");
  const loadingText = document.getElementById("loadingText");
  const newChatBtn = document.getElementById("newChatBtn");
  const exampleChips = document.querySelectorAll(".example-chip");

  // Toast References
  const toast = document.getElementById("toastNotification");
  const toastMessage = document.getElementById("toastMessage");
  const toastRetryBtn = document.getElementById("toastRetryBtn");
  const toastDismissBtn = document.getElementById("toastDismissBtn");

  // ==========================================================================
  // State
  // ==========================================================================
  let conversationHistory = [];
  let isProcessing = false;
  let lastFailedMessage = null;
  let toastTimer = null;

  // ==========================================================================
  // Markdown & Text Formatting Helpers
  // ==========================================================================
  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  function formatInlineMarkdown(text) {
    return text
      .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
      .replace(/\*(.*?)\*/g, "<em>$1</em>")
      .replace(/`([^`]+)`/g, "<code>$1</code>");
  }

  function formatMarkdown(text) {
    if (!text) return "";
    const lines = text.split("\n");
    const html = [];
    let inTable = false;
    let tableRows = [];
    let inList = false;

    const flushTable = () => {
      if (!tableRows.length) return;
      let tableHtml = '<div class="table-wrap"><table>';
      tableRows.forEach((r, idx) => {
        const isHeader = (idx === 0);
        const cells = r.split("|").slice(1, -1);
        if (cells.some(c => c.trim().match(/^:?-+:?$/))) return;
        tableHtml += "<tr>";
        cells.forEach(c => {
          const tag = isHeader ? "th" : "td";
          tableHtml += `<${tag}>${formatInlineMarkdown(c.trim())}</${tag}>`;
        });
        tableHtml += "</tr>";
      });
      tableHtml += "</table></div>";
      html.push(tableHtml);
      tableRows = [];
      inTable = false;
    };

    const flushList = () => {
      if (inList) {
        html.push("</ul>");
        inList = false;
      }
    };

    for (const line of lines) {
      const trimmed = line.trim();

      // Table line
      if (trimmed.startsWith("|") && trimmed.endsWith("|")) {
        flushList();
        inTable = true;
        tableRows.push(trimmed);
        continue;
      } else if (inTable) {
        flushTable();
      }

      // Unordered list item
      if (trimmed.startsWith("- ") || trimmed.startsWith("* ")) {
        if (!inList) {
          html.push("<ul>");
          inList = true;
        }
        html.push(`<li>${formatInlineMarkdown(trimmed.substring(2))}</li>`);
        continue;
      } else if (inList) {
        flushList();
      }

      // Headings
      if (trimmed.startsWith("#### ")) {
        html.push(`<h4>${formatInlineMarkdown(trimmed.substring(5))}</h4>`);
      } else if (trimmed.startsWith("### ")) {
        html.push(`<h3>${formatInlineMarkdown(trimmed.substring(4))}</h3>`);
      } else if (trimmed.startsWith("## ")) {
        html.push(`<h2>${formatInlineMarkdown(trimmed.substring(3))}</h2>`);
      } else if (trimmed.startsWith("# ")) {
        html.push(`<h1>${formatInlineMarkdown(trimmed.substring(2))}</h1>`);
      } else if (trimmed.length > 0) {
        html.push(`<p>${formatInlineMarkdown(trimmed)}</p>`);
      }
    }

    if (inTable) flushTable();
    if (inList) flushList();

    return html.join("");
  }

  // ==========================================================================
  // Auto-grow Input & Keyboard Handling
  // ==========================================================================
  chatInput.addEventListener("input", () => {
    chatInput.style.height = "auto";
    chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + "px";
  });

  chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!isProcessing && chatInput.value.trim().length > 0) {
        chatForm.requestSubmit();
      }
    }
  });

  // ==========================================================================
  // Chat Submission Logic
  // ==========================================================================
  chatForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const query = chatInput.value.trim();
    if (!query || isProcessing) return;

    chatInput.value = "";
    chatInput.style.height = "auto";
    submitUserQuery(query);
  });

  // Clickable example questions
  exampleChips.forEach((chip) => {
    chip.addEventListener("click", () => {
      const query = chip.dataset.query;
      if (query && !isProcessing) {
        submitUserQuery(query);
      }
    });
  });

  async function submitUserQuery(query) {
    if (isProcessing) return;
    isProcessing = true;
    lastFailedMessage = query;

    // Hide welcome screen immediately
    welcomeScreen.style.display = "none";
    hideToast();

    // Render user message bubble
    appendUserMessage(query);

    // Show loading feedback
    loadingIndicator.style.display = "flex";
    loadingText.textContent = "ORDIX is checking your order data...";
    scrollChatToBottom();

    // Disable send button while processing
    sendBtn.disabled = true;

    try {
      const payload = {
        message: query,
        conversation_history: conversationHistory.slice(-10),
      };

      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.detail || errorData.error || `Server responded with ${res.status}`);
      }

      const data = await res.json();
      const answer = data.answer || "I could not find an answer for that inquiry.";

      // Update history
      conversationHistory.push({ role: "user", content: query });
      conversationHistory.push({ role: "model", content: answer });

      // Render assistant message
      appendAssistantMessage(answer);
      lastFailedMessage = null;
    } catch (err) {
      console.error("Chat request failed:", err);
      const errMsg = err.message || "Failed to process query.";
      appendErrorMessage(errMsg, query);
      showToast(errMsg, true);
    } finally {
      isProcessing = false;
      loadingIndicator.style.display = "none";
      sendBtn.disabled = false;
      scrollChatToBottom();
      chatInput.focus();
    }
  }

  // ==========================================================================
  // Render Message Bubbles
  // ==========================================================================
  function appendUserMessage(text) {
    const row = document.createElement("div");
    row.className = "message-row user";
    row.innerHTML = `
      <div class="message-bubble">
        <p>${escapeHtml(text)}</p>
      </div>
    `;
    messagesList.appendChild(row);
    scrollChatToBottom();
  }

  function appendAssistantMessage(markdownText) {
    const row = document.createElement("div");
    row.className = "message-row assistant";
    row.innerHTML = `
      <div class="avatar-bot">O</div>
      <div class="message-bubble">
        ${formatMarkdown(markdownText)}
        <div class="message-meta">
          <span class="meta-badge">
            <span class="meta-dot"></span>
            orders.csv verified
          </span>
          <button class="btn-copy" title="Copy response to clipboard">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <rect width="14" height="14" x="8" y="8" rx="2" ry="2"></rect>
              <path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"></path>
            </svg>
            <span>Copy</span>
          </button>
        </div>
      </div>
    `;

    const copyBtn = row.querySelector(".btn-copy");
    copyBtn.addEventListener("click", () => {
      navigator.clipboard.writeText(markdownText).then(() => {
        copyBtn.innerHTML = `
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polyline points="20 6 9 17 4 12"></polyline>
          </svg>
          <span>Copied</span>
        `;
        setTimeout(() => {
          copyBtn.innerHTML = `
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <rect width="14" height="14" x="8" y="8" rx="2" ry="2"></rect>
              <path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"></path>
            </svg>
            <span>Copy</span>
          `;
        }, 1500);
      });
    });

    messagesList.appendChild(row);
    scrollChatToBottom();
  }

  function appendErrorMessage(errorMsg, failedQuery) {
    const row = document.createElement("div");
    row.className = "message-row assistant";
    row.innerHTML = `
      <div class="avatar-bot">O</div>
      <div class="message-bubble chat-error-card">
        <div class="chat-error-title">Unable to answer this query</div>
        <p>${escapeHtml(errorMsg)}</p>
        <button class="btn-retry-inline">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M21 12a9 9 0 0 0-9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"></path>
            <path d="M3 3v5h5"></path>
            <path d="M3 12a9 9 0 0 0 9 9 9.75 9.75 0 0 0 6.74-2.74L21 16"></path>
            <path d="M16 21h5v-5"></path>
          </svg>
          <span>Retry question</span>
        </button>
      </div>
    `;

    const retryBtn = row.querySelector(".btn-retry-inline");
    retryBtn.addEventListener("click", () => {
      row.remove();
      submitUserQuery(failedQuery);
    });

    messagesList.appendChild(row);
    scrollChatToBottom();
  }

  function scrollChatToBottom() {
    chatContainer.scrollTop = chatContainer.scrollHeight;
  }

  // ==========================================================================
  // New Chat Action
  // ==========================================================================
  function resetConversation() {
    conversationHistory = [];
    messagesList.innerHTML = "";
    welcomeScreen.style.display = "flex";
    loadingIndicator.style.display = "none";
    hideToast();
    chatInput.value = "";
    chatInput.style.height = "auto";
    chatInput.focus();
  }

  newChatBtn.addEventListener("click", resetConversation);

  // ==========================================================================
  // Toast Notification Helpers
  // ==========================================================================
  function showToast(msg, canRetry = false) {
    toastMessage.textContent = msg;
    toastRetryBtn.style.display = canRetry ? "inline-block" : "none";
    toast.style.display = "flex";
    clearTimeout(toastTimer);
    toastTimer = setTimeout(hideToast, 4000);
  }

  function hideToast() {
    toast.style.display = "none";
  }

  toastDismissBtn.addEventListener("click", hideToast);
  toastRetryBtn.addEventListener("click", () => {
    hideToast();
    if (lastFailedMessage) {
      submitUserQuery(lastFailedMessage);
    }
  });

  // ==========================================================================
  // Splash Screen Controller (Plain white background, fade-in, 2s reveal, once per session)
  // ==========================================================================
  const splashScreen = document.getElementById("splashScreen");
  if (splashScreen) {
    const alreadyShown = sessionStorage.getItem("ordix_splash_shown") === "true";
    if (alreadyShown) {
      splashScreen.style.display = "none";
      chatInput.focus();
    } else {
      // Hold for 2 seconds with smooth fade-in, then fade out into the chat interface
      const dismissSplash = () => {
        splashScreen.classList.add("fade-out");
        sessionStorage.setItem("ordix_splash_shown", "true");
        setTimeout(() => {
          splashScreen.style.display = "none";
          chatInput.focus();
        }, 700);
      };

      const splashTimer = setTimeout(dismissSplash, 2000);

      // Optional user click to skip immediately
      splashScreen.addEventListener("click", () => {
        clearTimeout(splashTimer);
        dismissSplash();
      });
    }
  } else {
    chatInput.focus();
  }
});
