document.addEventListener('DOMContentLoaded', () => {
    const messagesContainer = document.getElementById('messages');
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const typingIndicator = document.getElementById('typing-indicator');

    let currentBotMessage = null;

    function addUserMessage(text) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message user-message';
        messageDiv.textContent = text;
        messagesContainer.insertBefore(messageDiv, messagesContainer.firstChild);
    }

    function startNewBotMessage() {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message bot-message';
        messageDiv.textContent = '';
        messagesContainer.insertBefore(messageDiv, messagesContainer.firstChild);
        return messageDiv;
    }

    function showTypingIndicator() {
        typingIndicator.style.display = 'block';
    }

    function hideTypingIndicator() {
        typingIndicator.style.display = 'none';
    }

    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const message = userInput.value.trim();
        if (!message) return;

        // Clear input
        userInput.value = '';

        // Add user message
        addUserMessage(message);

        // Show typing indicator
        showTypingIndicator();

        // Start new bot message
        currentBotMessage = startNewBotMessage();

        // Start SSE connection
        const eventSource = new EventSource(`/chat?message=${encodeURIComponent(message)}`);

        eventSource.onmessage = (event) => {
            // Prepend new token to the message (since we're generating backwards)
            currentBotMessage.textContent = event.data + ' ' + currentBotMessage.textContent;
        };

        eventSource.onerror = () => {
            eventSource.close();
            hideTypingIndicator();
        };
    });
}); 