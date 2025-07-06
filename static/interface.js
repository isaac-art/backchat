document.addEventListener('DOMContentLoaded', () => {
    const messagesContainer = document.getElementById('messages');
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const typingIndicator = document.getElementById('typing-indicator');
    const saveToggle = document.getElementById('save-toggle');
    const sendButton = document.querySelector('button[type="submit"]');
    const toggleText = document.querySelector('.toggle-text');
    const savePreferenceKey = 'saveConversations';
    const settingsModal = document.getElementById('settings-modal');
    const settingsSaveToggle = document.getElementById('settings-save-toggle');
    const settingsToggleText = document.getElementById('settings-toggle-text');
    const privacyModal = document.getElementById('privacy-modal');
    const privacyAccept = document.getElementById('privacy-accept');
    const privacyDecline = document.getElementById('privacy-decline');

    // Get or create user ID from localStorage
    async function initializeUser() {
        let userId = localStorage.getItem('userId');
        if (!userId) {
            try {
                const response = await fetch('/user-id');
                const data = await response.json();
                userId = data.user_id;
                localStorage.setItem('userId', userId);
            } catch (error) {
                console.error('Error getting user ID:', error);
            }
        } else {
            // If we have an existing user ID, mark this as a new chat session
            try {
                await fetch('/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ 
                        message: "NEW_CHAT",
                        save_conversation: true,
                        user_id: userId,
                        is_new_chat: true
                    })
                });
            } catch (error) {
                console.error('Error marking new chat:', error);
            }
        }
        console.log('User ID:', userId);
    }
    initializeUser();

    // Remove old save toggle logic
    // Load save preference from localStorage, or show privacy modal if not set
    function showPrivacyModalIfNeeded() {
        const pref = localStorage.getItem(savePreferenceKey);
        if (pref === null) {
            privacyModal.style.display = 'block';
            document.body.style.overflow = 'hidden';
        } else {
            privacyModal.style.display = 'none';
            document.body.style.overflow = 'auto';
        }
    }
    showPrivacyModalIfNeeded();

    function setSavePreference(isSaving) {
        localStorage.setItem(savePreferenceKey, isSaving ? 'true' : 'false');
        updateSettingsToggle();
    }

    // Privacy modal button handlers
    privacyAccept.addEventListener('click', () => {
        setSavePreference(true);
        privacyModal.style.display = 'none';
        document.body.style.overflow = 'auto';
    });
    privacyDecline.addEventListener('click', () => {
        setSavePreference(false);
        privacyModal.style.display = 'none';
        document.body.style.overflow = 'auto';
    });

    // Settings modal logic
    function updateSettingsToggle() {
        if (!settingsSaveToggle || !settingsToggleText) return;
        const isSaving = localStorage.getItem(savePreferenceKey) === 'true';
        settingsSaveToggle.checked = isSaving;
        settingsToggleText.textContent = isSaving ? 'Saving conversations' : 'Not saving conversations';
        // Update visual state for custom toggle
        const toggleBox = document.querySelector('.settings-toggle-box');
        if (toggleBox) {
            toggleBox.classList.toggle('active', isSaving);
        }
        const savingLabel = document.getElementById('settings-saving-label');
        const notSavingLabel = document.getElementById('settings-not-saving-label');
        if (savingLabel && notSavingLabel) {
            if (isSaving) {
                savingLabel.classList.add('active');
                notSavingLabel.classList.remove('active');
            } else {
                savingLabel.classList.remove('active');
                notSavingLabel.classList.add('active');
            }
        }
    }
    if (settingsSaveToggle) {
        settingsSaveToggle.addEventListener('change', () => {
            setSavePreference(settingsSaveToggle.checked);
        });
    }
    updateSettingsToggle();

    // Health check monitoring
    async function checkHealth() {
        try {
            const response = await fetch('/health');
            const data = await response.json();
            
            if (data.status !== 'healthy') {
                sendButton.disabled = true;
                sendButton.style.backgroundColor = 'red';
                sendButton.style.opacity = '0.5';
                return;
            }

            if (data.demo_mode) {
                sendButton.disabled = false;
                sendButton.style.backgroundColor = '#ffa600'; // Orange color
                sendButton.style.opacity = '1';
            } else if (!data.model_loaded) {
                sendButton.disabled = true;
                sendButton.style.backgroundColor = 'red';
                sendButton.style.opacity = '0.5';
            } else {
                sendButton.disabled = false;
                sendButton.style.backgroundColor = 'var(--white)';
                sendButton.style.opacity = '1';
            }
        } catch (error) {
            console.error('Health check failed:', error);
            sendButton.disabled = true;
            sendButton.style.backgroundColor = 'red';
            sendButton.style.opacity = '0.5';
        }
    }

    // Check health every second
    setInterval(checkHealth, 1000);
    checkHealth(); // Initial check

    // Fullscreen functionality
    // document.addEventListener('keydown', (e) => {
    //     if (e && e.key && e.key.toLowerCase() === 'c') {
    //         if (!document.fullscreenElement) {
    //             document.documentElement.requestFullscreen({ navigationUI: "hide" })
    //                 .catch(err => console.error(`Error attempting to enable fullscreen: ${err.message}`));
    //         } else {
    //             document.exitFullscreen()
    //                 .catch(err => console.error(`Error attempting to exit fullscreen: ${err.message}`));
    //         }
    //     }
    // });

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
    let fullResponse = ''; // Track the full response for saving

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
        fullResponse = ''; // Reset full response
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
        fullResponse += cleanToken; // Add to full response
        
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

        // Get user ID from localStorage
        const userId = localStorage.getItem('userId');
        if (!userId) {
            console.error('No user ID found');
            return false;
        }

        // Clear input
        userInput.value = '';

        // Add user message
        addUserMessage(message);

        // Show typing indicator and start new bot message
        currentBotMessage = startNewBotMessage();
        showTypingIndicator();
        
        let reader = null;
        
        try {
            const savePref = localStorage.getItem(savePreferenceKey) === 'true';
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                    message,
                    save_conversation: savePref,
                    user_id: userId
                })
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

    function openModal(id) {
        document.getElementById(`${id}-modal`).style.display = 'block';
        document.body.style.overflow = 'hidden';
        if (id === 'settings') {
            updateSettingsToggle();
        }
    }

    // Ensure settings toggle is in sync with localStorage on page load
    updateSettingsToggle();
}); 