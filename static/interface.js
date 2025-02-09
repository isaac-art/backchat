document.addEventListener('DOMContentLoaded', () => {
    const messagesContainer = document.getElementById('messages');
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const typingIndicator = document.getElementById('typing-indicator');

    let currentBotMessage = null;
    let lastWord = null;  // Track the last word we received

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
        lastWord = null;  // Reset word tracking
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

    function isPunctuation(token) {
        // Updated to include 's' and 'ies' and handle whitespace
        const cleanToken = token.trim();
        return /^[\s]*[.,!?:;"')\]}]+[\s]*$/.test(cleanToken) || 
               cleanToken === 's' || 
               cleanToken === 'ies';
    }

    function hasPunctuationEnd(token) {
        return /[.,!?:;]$/.test(token);
    }

    function extractPunctuation(token) {
        const match = token.match(/([^.,!?:;]+)([.,!?:;]+)$/);
        if (match) {
            return [match[1], match[2]];  // [word, punctuation]
        }
        return [token, ''];  // no punctuation
    }

    function handleNewToken(token) {
        const cleanToken = token.trim();
        
        if (!currentBotMessage.textContent) {
            // First token case
            currentBotMessage.textContent = cleanToken;
            lastWord = cleanToken;
            return;
        }

        if (isPunctuation(cleanToken)) {
            // For punctuation, attach it to the last word
            const words = currentBotMessage.textContent.split(' ').filter(w => w);
            if (words.length > 0) {
                // Attach punctuation to the first word (since text is reversed)
                words[0] = words[0] + cleanToken;
                currentBotMessage.textContent = words.join(' ');
            }
        } else {
            // For regular words, prepend with space
            currentBotMessage.textContent = cleanToken + ' ' + currentBotMessage.textContent;
            lastWord = cleanToken;
        }
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
                        handleNewToken(token);
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