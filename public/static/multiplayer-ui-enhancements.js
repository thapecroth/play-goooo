// UI Enhancement Module for Multiplayer Go
(function() {
    'use strict';

    // Add UI enhancements when DOM is loaded
    document.addEventListener('DOMContentLoaded', function() {
        enhanceUI();
        addAnimations();
        improveAccessibility();
    });

    function enhanceUI() {
        // Add ripple effect to buttons
        const buttons = document.querySelectorAll('.btn');
        buttons.forEach(button => {
            button.addEventListener('click', createRipple);
        });

        // Enhanced room code input
        const roomCodeInput = document.getElementById('room-code');
        if (roomCodeInput) {
            roomCodeInput.addEventListener('input', function(e) {
                // Auto-uppercase and format
                e.target.value = e.target.value.toUpperCase().replace(/[^A-Z0-9]/g, '');
                
                // Auto-submit when 6 characters entered
                if (e.target.value.length === 6) {
                    const joinBtn = document.getElementById('join-room');
                    joinBtn.classList.add('pulse');
                    setTimeout(() => joinBtn.classList.remove('pulse'), 1000);
                }
            });

            // Add visual feedback for valid/invalid codes
            roomCodeInput.addEventListener('input', function(e) {
                if (e.target.value.length === 6) {
                    e.target.style.borderColor = 'var(--success)';
                } else if (e.target.value.length > 0) {
                    e.target.style.borderColor = 'var(--warning)';
                } else {
                    e.target.style.borderColor = '';
                }
            });
        }

        // Enhanced player name input
        const playerNameInput = document.getElementById('player-name');
        if (playerNameInput) {
            // Add character counter
            const charCounter = document.createElement('div');
            charCounter.className = 'char-counter';
            charCounter.style.cssText = 'text-align: center; margin-top: 8px; font-size: 12px; color: var(--gray-500);';
            playerNameInput.parentNode.appendChild(charCounter);

            playerNameInput.addEventListener('input', function(e) {
                const remaining = 20 - e.target.value.length;
                charCounter.textContent = `${remaining} characters remaining`;
                charCounter.style.color = remaining < 5 ? 'var(--warning)' : 'var(--gray-500)';
            });
        }

        // Add loading states to buttons
        const actionButtons = document.querySelectorAll('#create-room, #join-room');
        actionButtons.forEach(button => {
            const originalText = button.textContent;
            button.addEventListener('click', function() {
                if (!button.disabled) {
                    button.disabled = true;
                    button.innerHTML = '<span class="loading-spinner"></span> Loading...';
                    // Re-enable after 3 seconds (or when server responds)
                    setTimeout(() => {
                        button.disabled = false;
                        button.textContent = originalText;
                    }, 3000);
                }
            });
        });

        // Enhance room list items
        observeRoomList();

        // Add tooltips
        addTooltips();

        // Enhance chat experience
        enhanceChat();
    }

    function createRipple(e) {
        const button = e.currentTarget;
        const ripple = document.createElement('span');
        const diameter = Math.max(button.clientWidth, button.clientHeight);
        const radius = diameter / 2;

        ripple.style.width = ripple.style.height = diameter + 'px';
        ripple.style.left = e.clientX - button.offsetLeft - radius + 'px';
        ripple.style.top = e.clientY - button.offsetTop - radius + 'px';
        ripple.classList.add('ripple');

        // Remove any existing ripples
        const existingRipple = button.getElementsByClassName('ripple')[0];
        if (existingRipple) {
            existingRipple.remove();
        }

        button.appendChild(ripple);
    }

    function observeRoomList() {
        const roomsList = document.getElementById('rooms-list');
        if (!roomsList) return;

        // Observe changes to room list
        const observer = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                mutation.addedNodes.forEach((node) => {
                    if (node.classList && node.classList.contains('room-item')) {
                        // Add entrance animation
                        node.style.animation = 'roomSlideIn 0.4s ease';
                        
                        // Add hover sound effect
                        node.addEventListener('mouseenter', () => {
                            if (window.gameSounds && window.gameSounds.enabled) {
                                // Create a subtle hover sound
                                const audioContext = new (window.AudioContext || window.webkitAudioContext)();
                                const oscillator = audioContext.createOscillator();
                                const gainNode = audioContext.createGain();
                                
                                oscillator.connect(gainNode);
                                gainNode.connect(audioContext.destination);
                                
                                oscillator.frequency.value = 800;
                                gainNode.gain.value = 0.05;
                                
                                oscillator.start();
                                oscillator.stop(audioContext.currentTime + 0.05);
                            }
                        });
                    }
                });
            });
        });

        observer.observe(roomsList, { childList: true });
    }

    function addTooltips() {
        const tooltipElements = [
            { el: document.getElementById('copy-room-code'), text: 'Copy room code to clipboard' },
            { el: document.getElementById('pass'), text: 'Skip your turn' },
            { el: document.getElementById('resign'), text: 'Forfeit the game' },
            { el: document.getElementById('sound-toggle'), text: 'Toggle sound effects on/off' }
        ];

        tooltipElements.forEach(({ el, text }) => {
            if (el) {
                el.setAttribute('data-tooltip', text);
                el.classList.add('has-tooltip');
            }
        });
    }

    function enhanceChat() {
        const chatInput = document.getElementById('chat-input');
        const chatMessages = document.getElementById('chat-messages');
        
        if (!chatInput || !chatMessages) return;

        // Add emoji picker button
        const emojiButton = document.createElement('button');
        emojiButton.className = 'btn btn-small emoji-picker';
        emojiButton.textContent = '😊';
        emojiButton.title = 'Add emoji';
        emojiButton.style.cssText = 'margin-left: 8px;';
        
        chatInput.parentNode.insertBefore(emojiButton, chatInput.nextSibling);

        // Simple emoji panel
        const emojis = ['😊', '👍', '👏', '🎯', '🤔', '😅', '🙏', '💪', '🎉', '😮'];
        const emojiPanel = document.createElement('div');
        emojiPanel.className = 'emoji-panel';
        emojiPanel.style.cssText = `
            position: absolute;
            bottom: 100%;
            right: 0;
            background: white;
            border: 2px solid var(--gray-300);
            border-radius: 12px;
            padding: 8px;
            display: none;
            grid-template-columns: repeat(5, 1fr);
            gap: 4px;
            box-shadow: var(--shadow-lg);
            z-index: 1000;
        `;

        emojis.forEach(emoji => {
            const emojiBtn = document.createElement('button');
            emojiBtn.textContent = emoji;
            emojiBtn.className = 'emoji-btn';
            emojiBtn.style.cssText = `
                font-size: 24px;
                padding: 8px;
                border: none;
                background: transparent;
                cursor: pointer;
                border-radius: 8px;
                transition: all 0.2s;
            `;
            emojiBtn.addEventListener('click', () => {
                chatInput.value += emoji;
                chatInput.focus();
                emojiPanel.style.display = 'none';
            });
            emojiBtn.addEventListener('mouseenter', () => {
                emojiBtn.style.background = 'var(--gray-100)';
                emojiBtn.style.transform = 'scale(1.2)';
            });
            emojiBtn.addEventListener('mouseleave', () => {
                emojiBtn.style.background = 'transparent';
                emojiBtn.style.transform = 'scale(1)';
            });
            emojiPanel.appendChild(emojiBtn);
        });

        emojiButton.style.position = 'relative';
        emojiButton.appendChild(emojiPanel);

        emojiButton.addEventListener('click', (e) => {
            e.preventDefault();
            emojiPanel.style.display = emojiPanel.style.display === 'grid' ? 'none' : 'grid';
        });

        // Close emoji panel when clicking outside
        document.addEventListener('click', (e) => {
            if (!emojiButton.contains(e.target)) {
                emojiPanel.style.display = 'none';
            }
        });

        // Auto-scroll chat to bottom on new messages
        const chatObserver = new MutationObserver(() => {
            chatMessages.scrollTop = chatMessages.scrollHeight;
        });
        chatObserver.observe(chatMessages, { childList: true });
    }

    function addAnimations() {
        // Add CSS for ripple effect
        const style = document.createElement('style');
        style.textContent = `
            .ripple {
                position: absolute;
                border-radius: 50%;
                background: rgba(255, 255, 255, 0.6);
                transform: scale(0);
                animation: ripple-animation 0.6s ease-out;
                pointer-events: none;
            }

            @keyframes ripple-animation {
                to {
                    transform: scale(4);
                    opacity: 0;
                }
            }

            .pulse {
                animation: pulse-animation 1s ease-in-out;
            }

            @keyframes pulse-animation {
                0%, 100% { transform: scale(1); }
                50% { transform: scale(1.05); }
            }

            @keyframes roomSlideIn {
                from {
                    opacity: 0;
                    transform: translateY(20px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }

            .loading-spinner {
                display: inline-block;
                width: 16px;
                height: 16px;
                border: 2px solid rgba(255, 255, 255, 0.3);
                border-top-color: white;
                border-radius: 50%;
                animation: spin 0.8s linear infinite;
            }

            /* Tooltip styles */
            .has-tooltip {
                position: relative;
            }

            .has-tooltip::after {
                content: attr(data-tooltip);
                position: absolute;
                bottom: 100%;
                left: 50%;
                transform: translateX(-50%) translateY(-8px);
                background: var(--gray-800);
                color: white;
                padding: 6px 12px;
                border-radius: 6px;
                font-size: 12px;
                white-space: nowrap;
                opacity: 0;
                pointer-events: none;
                transition: opacity 0.3s, transform 0.3s;
            }

            .has-tooltip::before {
                content: '';
                position: absolute;
                bottom: 100%;
                left: 50%;
                transform: translateX(-50%);
                border: 6px solid transparent;
                border-top-color: var(--gray-800);
                opacity: 0;
                transition: opacity 0.3s;
            }

            .has-tooltip:hover::after,
            .has-tooltip:hover::before {
                opacity: 1;
                transform: translateX(-50%) translateY(-12px);
            }

            /* Connection status enhancements */
            @keyframes connectionPulse {
                0%, 100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.4); }
                50% { box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
            }

            .connection-status.online {
                animation: connectionPulse 3s infinite;
            }

            /* Score card animations */
            .score {
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            }

            .score.active {
                animation: activePlayerPulse 2s ease-in-out infinite;
            }

            @keyframes activePlayerPulse {
                0%, 100% { 
                    transform: scale(1);
                    box-shadow: var(--shadow-lg);
                }
                50% { 
                    transform: scale(1.02);
                    box-shadow: 0 12px 24px rgba(99, 102, 241, 0.3);
                }
            }

            /* Enhanced focus states */
            input:focus, select:focus, button:focus {
                animation: focusPulse 0.5s ease;
            }

            @keyframes focusPulse {
                0% { transform: scale(1); }
                50% { transform: scale(1.02); }
                100% { transform: scale(1); }
            }
        `;
        document.head.appendChild(style);
    }

    function improveAccessibility() {
        // Add ARIA labels
        const ariaLabels = [
            { id: 'create-room', label: 'Create a new game room' },
            { id: 'join-room', label: 'Join an existing game room' },
            { id: 'room-code', label: 'Enter 6-character room code' },
            { id: 'player-name', label: 'Enter your player name' },
            { id: 'new-board-size', label: 'Select board size for new game' },
            { id: 'vs-ai', label: 'Toggle play against AI' },
            { id: 'pass', label: 'Pass your turn' },
            { id: 'resign', label: 'Resign from the game' },
            { id: 'chat-input', label: 'Type a chat message' },
            { id: 'send-chat', label: 'Send chat message' }
        ];

        ariaLabels.forEach(({ id, label }) => {
            const element = document.getElementById(id);
            if (element) {
                element.setAttribute('aria-label', label);
            }
        });

        // Add keyboard navigation improvements
        const roomsList = document.getElementById('rooms-list');
        if (roomsList) {
            roomsList.setAttribute('role', 'list');
            roomsList.setAttribute('aria-label', 'Available game rooms');
        }

        // Improve form semantics
        const forms = document.querySelectorAll('.create-game-section, .join-game-section');
        forms.forEach(form => {
            form.setAttribute('role', 'form');
        });
    }

    // Export for use in other scripts
    window.multiplayerUIEnhancements = {
        showNotification: function(message, type = 'info') {
            const notification = document.createElement('div');
            notification.className = `notification notification-${type}`;
            notification.textContent = message;
            notification.style.cssText = `
                position: fixed;
                top: 20px;
                right: 20px;
                padding: 16px 24px;
                background: white;
                border-radius: 12px;
                box-shadow: var(--shadow-xl);
                z-index: 9999;
                animation: notificationSlide 0.3s ease;
                max-width: 300px;
            `;

            const style = document.createElement('style');
            style.textContent = `
                @keyframes notificationSlide {
                    from {
                        opacity: 0;
                        transform: translateX(100%);
                    }
                    to {
                        opacity: 1;
                        transform: translateX(0);
                    }
                }
            `;
            document.head.appendChild(style);

            document.body.appendChild(notification);

            setTimeout(() => {
                notification.style.animation = 'notificationSlide 0.3s ease reverse';
                setTimeout(() => notification.remove(), 300);
            }, 3000);
        }
    };
})();