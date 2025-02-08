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
        messageDiv.scrollIntoView({ behavior: 'smooth' });
    }

    function startNewBotMessage() {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message bot-message';
        messageDiv.textContent = '';
        messagesContainer.insertBefore(messageDiv, messagesContainer.firstChild);
        messageDiv.scrollIntoView({ behavior: 'smooth' });
        return messageDiv;
    }

    function showTypingIndicator() {
        typingIndicator.style.display = 'block';
        messagesContainer.insertBefore(typingIndicator, messagesContainer.firstChild);
        typingIndicator.scrollIntoView({ behavior: 'smooth' });
    }

    function hideTypingIndicator() {
        typingIndicator.style.display = 'none';
    }

    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        e.stopPropagation();
        
        const message = userInput.value.trim();
        if (!message) return false;

        // Clear input
        userInput.value = '';

        // Add user message
        addUserMessage(message);

        // Show typing indicator and start new bot message
        currentBotMessage = startNewBotMessage();
        showTypingIndicator();
        
        // Move the user message after the bot message container
        // const userMessage = messagesContainer.lastElementChild.previousElementSibling;
        // messagesContainer.insertBefore(userMessage, currentBotMessage);

        try {
            // Create POST request for server-sent events
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;
                
                const chunk = decoder.decode(value);
                const lines = chunk.split('\n');
                
                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const token = line.slice(6);
                        // Prepend new token to the message (since we're generating backwards)
                        currentBotMessage.textContent = token + ' ' + currentBotMessage.textContent;
                    }
                }
            }
        } catch (error) {
            console.error('Error:', error);
            currentBotMessage.textContent = 'Error: Failed to get response';
        } finally {
            hideTypingIndicator();
            if (reader) reader.releaseLock();
        }
        
        return false;
    });
}); 