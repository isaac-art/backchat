document.addEventListener('DOMContentLoaded', () => {
    const messagesContainer = document.getElementById('messages');
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const typingIndicator = document.getElementById('typing-indicator');

    // Fullscreen functionality
    document.addEventListener('keydown', (e) => {
        if (e.key.toLowerCase() === 'c') {
            if (!document.fullscreenElement) {
                document.documentElement.requestFullscreen({ navigationUI: "hide" })
                    .catch(err => console.error(`Error attempting to enable fullscreen: ${err.message}`));
            } else {
                document.exitFullscreen()
                    .catch(err => console.error(`Error attempting to exit fullscreen: ${err.message}`));
            }
        }
    });

    // Create ambient lights container
    const ambientContainer = document.createElement('div');
    ambientContainer.className = 'ambient-container';
    document.body.appendChild(ambientContainer);

    // Function to create ambient lights
    function createAmbientLight() {
        const light = document.createElement('div');
        light.className = 'ambient-light';
        
        // Random starting position
        const startX = Math.random() * window.innerWidth;
        const startY = Math.random() * window.innerHeight;
        light.style.left = `${startX}px`;
        light.style.top = `${startY}px`;
        
        // Random movement
        const moveX = (Math.random() - 0.5) * 200;
        const moveY = (Math.random() - 0.5) * 200;
        light.style.setProperty('--move-x', `${moveX}px`);
        light.style.setProperty('--move-y', `${moveY}px`);
        
        ambientContainer.appendChild(light);
        
        // Remove light after animation
        light.addEventListener('animationend', () => {
            light.remove();
        });
    }

    // Create new ambient light occasionally
    setInterval(createAmbientLight, 2000);  // More frequent creation

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
        
        let reader = null;
        
        try {
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

            reader = response.body.getReader();
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
            if (reader) {
                try {
                    await reader.releaseLock();
                } catch (e) {
                    console.error('Error releasing reader lock:', e);
                }
            }
        }
        
        return false;
    });
}); 