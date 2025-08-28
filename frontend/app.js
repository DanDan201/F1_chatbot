const API_BASE = (localStorage.getItem('api_base') || 'http://localhost:5001');

const chatEl = document.getElementById('chat');
const formEl = document.getElementById('chat-form');
const inputEl = document.getElementById('user-input');

function addMsg(text, who = 'bot'){
    const div = document.createElement('div');
    div.className = `msg ${who}`;
    div.textContent = text;
    chatEl.appendChild(div);
    chatEl.scrollTop = chatEl.scrollHeight;
}

formEl.addEventListener('submit', async (e) => {
    e.preventDefault();
    const q = inputEl.value.trim();
    if (!q) return;
    addMsg(q, 'user');
    inputEl.value = '';
    addMsg('Thinking...');

    try {
        const res = await fetch(`${API_BASE}/chat`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ message: q})
        });
        const data = await res.json();
        const last = chatEl.querySelector('.msg.bot:last-child');
        last.textContent = data.answer || data.error || 'Error';
    }
    catch (err){
        const last = chatEl.querySelector('.msg.bot:last-child');
        last.textContent = 'Network Error';
    }
});