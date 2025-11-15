// Configuration
const API_BASE_URL = 'http://localhost:8000';

// DOM Elements
const chatContainer = document.getElementById('chatContainer');
const messageInput = document.getElementById('messageInput');
const sendButton = document.getElementById('sendButton');
const typingIndicator = document.getElementById('typingIndicator');

// State
let currentMessageId = null;
let ratings = {}; // Store ratings by message ID

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    // Auto-resize textarea
    messageInput.addEventListener('input', () => {
        messageInput.style.height = 'auto';
        messageInput.style.height = messageInput.scrollHeight + 'px';
    });
 
    // Send on Enter (Shift+Enter for new line)
    messageInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Send button click
    sendButton.addEventListener('click', sendMessage);

    // Example question click handlers
    document.querySelectorAll('.welcome-message li').forEach(li => {
        li.addEventListener('click', () => {
            messageInput.value = li.textContent.trim();
            sendMessage();
        });
    });
  
    // Check API health
    checkHealth();
});

async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();
        console.log('API Health:', data);
    } catch (error) {
        console.error('API Health Check Failed:', error);
        showError('Unable to connect to the API. Please make sure the server is running.');
    }
}

async function sendMessage() {
    const question = messageInput.value.trim();
    if (!question) return;

    // Clear input
    messageInput.value = '';
    messageInput.style.height = 'auto';

    // Remove welcome message
    const welcomeMsg = document.querySelector('.welcome-message');
    if (welcomeMsg) {
        welcomeMsg.remove();
    }

    // Add user message
    addMessage('user', question);

    // Show typing indicator
    showTypingIndicator();

    // Disable send button
    sendButton.disabled = true;

    try {
        const response = await fetch(`${API_BASE_URL}/api/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                question: question,
                top_k: 5
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        // Hide typing indicator
        hideTypingIndicator();

        // Add assistant message
        const messageId = addAssistantMessage(data, question);

        // Store question for rating
        ratings[messageId] = {
            question: question,
            answer: data.answer
        };

    } catch (error) {
        console.error('Error:', error);
        hideTypingIndicator();
        addMessage('assistant', `Sorry, I encountered an error: ${error.message}. Please try again.`, true);
    } finally {
        sendButton.disabled = false;
        messageInput.focus();
    }
}

function addMessage(role, content, isError = false) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';

    if (isError) {
        contentDiv.classList.add('error-message');
    }

    // Format content (simple markdown-like formatting)
    contentDiv.innerHTML = formatMessage(content);

    messageDiv.appendChild(contentDiv);
    chatContainer.appendChild(messageDiv);
    scrollToBottom();

    return messageDiv;
}

function addAssistantMessage(data, question) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';
    const messageId = `msg-${Date.now()}`;
    messageDiv.id = messageId;

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';

    // Main answer
    let html = `<div class="answer-content">${formatMessage(data.answer)}</div>`;

    // Reasoning section
    if (data.reasoning) {
        html += `
            <div class="reasoning-section">
                <strong>📊 Reasoning:</strong> ${data.reasoning}
            </div>
        `;
    }

    // Sources section
    if (data.sources && data.sources.length > 0) {
        html += `
            <div class="sources-section">
                <h4>📚 Sources (${data.sources.length}):</h4>
        `;

        data.sources.forEach((source, index) => {
            const sourceTypeClass = source.source_type === 'vector_embedding' ? 'vector' : 'kg';
            const sourceTypeLabel = source.source_type === 'vector_embedding' ? 'Vector' : 'KG';

            html += `
                <div class="source-item">
                    <span class="source-type ${sourceTypeClass}">${sourceTypeLabel}</span>
                    ${source.supplement_name ? `<strong>${source.supplement_name}</strong>` : ''}
                    ${source.section ? ` - ${source.section}` : ''}
                    ${source.score ? ` <small>(Score: ${source.score.toFixed(3)})</small>` : ''}
                </div>
            `;
        });

        html += `</div>`;
    }

    // Precaution notice
    if (data.precaution_notice) {
        html += `
            <div class="precaution-notice">
                ⚠️ <strong>Important:</strong> ${data.precaution_notice}
            </div>
        `;
    }

    // LLM knowledge indicator
    if (data.used_llm_knowledge) {
        html += `
            <div class="sources-section" style="background: #fff3cd; border-left: 4px solid #ffc107;">
                <small>ℹ️ Note: Some information may come from general knowledge as the database had limited information on this topic.</small>
            </div>
        `;
    }

    contentDiv.innerHTML = html;

    // Rating buttons and accuracy flag
    const actionsDiv = document.createElement('div');
    actionsDiv.className = 'message-actions';
    actionsDiv.innerHTML = `
        <button class="rating-button up" onclick="rateMessage('${messageId}', 1)" title="Helpful">
            👍 Helpful
        </button>
        <button class="rating-button down" onclick="rateMessage('${messageId}', -1)" title="Not helpful">
            👎 Not helpful
        </button>
        <button class="accuracy-flag-button" onclick="flagInaccurate('${messageId}')" title="Report inaccurate answer">
            🚩 Report Inaccurate
        </button>
    `;

    messageDiv.appendChild(contentDiv);
    messageDiv.appendChild(actionsDiv);
    chatContainer.appendChild(messageDiv);
    scrollToBottom();

    return messageId;
}

function formatMessage(text) {
    // Simple formatting: convert markdown-like syntax to HTML
    let html = text
        // Bold
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        // Italic
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        // Code
        .replace(/`(.*?)`/g, '<code>$1</code>')
        // Line breaks
        .replace(/\n/g, '<br>')
        // Bullet points
        .replace(/^[-•]\s+(.+)$/gm, '<li>$1</li>')
        // Numbered lists
        .replace(/^\d+\.\s+(.+)$/gm, '<li>$1</li>');

    // Wrap consecutive <li> in <ul>
    html = html.replace(/(<li>.*?<\/li>)/gs, (match) => {
        if (!match.includes('<ul>')) {
            return '<ul>' + match + '</ul>';
        }
        return match;
    });

    return html;
}

function showTypingIndicator() {
    typingIndicator.style.display = 'flex';
    scrollToBottom();
}

function hideTypingIndicator() {
    typingIndicator.style.display = 'none';
}

function scrollToBottom() {
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

async function rateMessage(messageId, rating) {
    const ratingData = ratings[messageId];
    if (!ratingData) {
        console.error('No rating data found for message:', messageId);
        return;
    }

    // Update UI
    const messageDiv = document.getElementById(messageId);
    if (messageDiv) {
        const buttons = messageDiv.querySelectorAll('.rating-button');
        buttons.forEach(btn => {
            btn.classList.remove('active');
        });

        if (rating === 1) {
            buttons[0].classList.add('active', 'up');
        } else {
            buttons[1].classList.add('active', 'down');
        }
    }

    try {
        const response = await fetch(`${API_BASE_URL}/api/rate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                question: ratingData.question,
                answer: ratingData.answer,
                rating: rating,
                feedback: null
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        console.log('Rating saved:', data);

        // Show thank you message (optional)
        if (messageDiv) {
            const actionsDiv = messageDiv.querySelector('.message-actions');
            if (actionsDiv) {
                const thankYou = document.createElement('small');
                thankYou.textContent = '✓ Thank you for your feedback!';
                thankYou.style.color = '#4caf50';
                thankYou.style.marginLeft = '10px';
                actionsDiv.appendChild(thankYou);
                setTimeout(() => thankYou.remove(), 3000);
            }
        }

    } catch (error) {
        console.error('Error rating message:', error);
        alert('Failed to save rating. Please try again.');
    }
}

async function flagInaccurate(messageId) {
    const ratingData = ratings[messageId];
    if (!ratingData) {
        console.error('No rating data found for message:', messageId);
        return;
    }

    // Ask for reason (optional)
    const reason = prompt('Please provide a reason why this answer is inaccurate (optional):');
    
    // Update UI
    const messageDiv = document.getElementById(messageId);
    if (messageDiv) {
        const flagButton = messageDiv.querySelector('.accuracy-flag-button');
        if (flagButton) {
            flagButton.classList.add('active');
            flagButton.disabled = true;
            flagButton.textContent = '🚩 Reported';
        }
    }

    try {
        const response = await fetch(`${API_BASE_URL}/api/report-accuracy`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                question: ratingData.question,
                answer: ratingData.answer,
                reason: reason || null
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        console.log('Accuracy report saved:', data);

        // Show thank you message
        if (messageDiv) {
            const actionsDiv = messageDiv.querySelector('.message-actions');
            if (actionsDiv) {
                const thankYou = document.createElement('small');
                thankYou.textContent = '✓ Thank you for reporting this!';
                thankYou.style.color = '#ffcc00';
                thankYou.style.marginLeft = '10px';
                actionsDiv.appendChild(thankYou);
                setTimeout(() => thankYou.remove(), 3000);
            }
        }

    } catch (error) {
        console.error('Error reporting accuracy:', error);
        alert('Failed to report inaccuracy. Please try again.');
        
        // Re-enable button on error
        if (messageDiv) {
            const flagButton = messageDiv.querySelector('.accuracy-flag-button');
            if (flagButton) {
                flagButton.classList.remove('active');
                flagButton.disabled = false;
                flagButton.textContent = '🚩 Report Inaccurate';
            }
        }
    }
}

// Make rateMessage and flagInaccurate available globally
window.rateMessage = rateMessage;
window.flagInaccurate = flagInaccurate;

