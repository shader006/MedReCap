const API = 'http://localhost:8888';
const MAX_ATTACHMENTS = 12;
const MAX_TEXT_FILE_BYTES = 16000;
const MAX_BINARY_ATTACHMENT_BYTES = 25 * 1024 * 1024;
const DOCLING_EXTENSIONS = [
    '.pdf', '.docx', '.doc', '.docm', '.rtf', '.odt',
    '.pptx', '.ppt', '.pptm', '.odp',
    '.xlsx', '.xls', '.xlsm', '.xlsb', '.ods',
    '.html', '.htm', '.md', '.csv', '.xml'
];

let history = [];
let attachments = [];
let isThinking = false;
let isLoading = false;
let isPreparingAttachments = false;

const messagesEl = document.getElementById('messages');
const inputEl = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');
const thinkToggle = document.getElementById('thinkingToggle');
const tempSlider = document.getElementById('tempSlider');
const tempVal = document.getElementById('tempVal');
const statusDot = document.getElementById('statusDot').querySelector('.dot');
const statusText = document.getElementById('statusText');
const modelStatus = document.getElementById('modelStatus');
const modelBadge = document.getElementById('modelBadge');
const attachmentTray = document.getElementById('attachmentTray');
const fileInput = document.getElementById('fileInput');
const folderInput = document.getElementById('folderInput');
const imageInput = document.getElementById('imageInput');
const composerAttachButtons = document.querySelectorAll('.composer-attach-btn');
const summaryBtn = document.getElementById('summaryBtn');

checkHealth();
setInterval(checkHealth, 30000);
fetchGpuStats();
setInterval(fetchGpuStats, 3000);
setComposerState();

tempSlider.addEventListener('input', () => {
    tempVal.textContent = (tempSlider.value / 10).toFixed(1);
});

thinkToggle.addEventListener('change', () => {
    isThinking = thinkToggle.checked;
});

inputEl.addEventListener('input', () => {
    inputEl.style.height = 'auto';
    inputEl.style.height = inputEl.scrollHeight + 'px';
});

inputEl.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

document.getElementById('newChatBtn').addEventListener('click', () => {
    history = [];
    clearAttachments();
    messagesEl.innerHTML = `
    <div class="welcome">
      <div class="welcome-icon">AI</div>
      <h2>Xin chao! Toi la Qwen3.5-2B</h2>
      <p>Ban co the gui them file, folder, va anh de dua them context vao cuoc hoi thoai.</p>
      <div class="suggestions">
        <button class="suggest-btn" onclick="suggest('Doc file dinh kem va tom tat')">Tom tat file</button>
        <button class="suggest-btn" onclick="suggest('So sanh cac file trong folder dinh kem')">So sanh folder</button>
        <button class="suggest-btn" onclick="suggest('Giai thich source code trong cac tep dinh kem')">Doc source code</button>
      </div>
    </div>`;
});

fileInput.addEventListener('change', (e) => handleSelectedFiles(e.target.files, 'file', e.target));
folderInput.addEventListener('change', (e) => handleSelectedFiles(e.target.files, 'folder', e.target));
imageInput.addEventListener('change', (e) => handleSelectedFiles(e.target.files, 'image', e.target));

async function checkHealth() {
    try {
        const r = await fetch(`${API}/health`);
        const data = await r.json();
        setStatus(data.model_ready ? 'online' : 'loading', data.model_ready ? 'Model san sang' : 'Dang tai model...');
        modelStatus.textContent = data.model_ready ? data.model : 'Chua san sang';
        modelBadge.textContent = data.model_ready ? data.model : 'Model unavailable';
    } catch {
        setStatus('error', 'Mat ket noi backend');
        modelStatus.textContent = 'Offline';
        modelBadge.textContent = 'Backend offline';
    }
}

function setStatus(type, text) {
    statusDot.className = 'dot ' + (type === 'online' ? 'online' : type === 'error' ? 'error' : '');
    statusText.textContent = text;
}

async function fetchGpuStats() {
    try {
        const r = await fetch(`${API}/gpu`);
        const data = await r.json();
        if (!data.available || !data.gpus?.length) {
            resetGpuStats();
            return;
        }

        const gpu = data.gpus[0];
        document.getElementById('gpuSection').style.display = 'block';
        document.getElementById('gpuName').textContent = gpu.name;
        document.getElementById('gpuUtil').textContent = `${gpu.utilization}%`;
        document.getElementById('gpuUtilBar').style.width = `${gpu.utilization}%`;

        const usedGB = (gpu.memory_used / 1024).toFixed(1);
        const totalGB = (gpu.memory_total / 1024).toFixed(1);
        document.getElementById('gpuMem').textContent = `${usedGB}/${totalGB} GB`;
        document.getElementById('gpuMemBar').style.width = `${gpu.memory_percent}%`;

        const tempEl = document.getElementById('gpuTemp');
        const tempRow = tempEl.closest('.temp-row');
        tempEl.textContent = `${gpu.temperature} C`;
        tempRow.className = 'gpu-row temp-row' +
            (gpu.temperature >= 85 ? ' danger' : gpu.temperature >= 70 ? ' hot' : '');
    } catch {
        resetGpuStats();
    }
}

function resetGpuStats() {
    document.getElementById('gpuSection').style.display = 'none';
    document.getElementById('gpuName').textContent = '-';
    document.getElementById('gpuUtil').textContent = '-%';
    document.getElementById('gpuUtilBar').style.width = '0%';
    document.getElementById('gpuMem').textContent = '-';
    document.getElementById('gpuMemBar').style.width = '0%';
    const tempEl = document.getElementById('gpuTemp');
    tempEl.textContent = '- C';
    tempEl.closest('.temp-row').className = 'gpu-row temp-row';
}

function suggest(text) {
    inputEl.value = text;
    sendMessage();
}

function openPicker(kind) {
    if (kind === 'file') fileInput.click();
    if (kind === 'folder') folderInput.click();
    if (kind === 'image') imageInput.click();
}

async function handleSelectedFiles(fileList, source, input) {
    const files = Array.from(fileList || []);
    input.value = '';

    if (!files.length) return;

    const remaining = MAX_ATTACHMENTS - attachments.length;
    const selected = files.slice(0, Math.max(remaining, 0));

    if (!selected.length) {
        setStatus('error', `Da dat gioi han ${MAX_ATTACHMENTS} tep dinh kem`);
        return;
    }

    isPreparingAttachments = true;
    setStatus('loading', 'Dang xu ly tep dinh kem...');
    setComposerState();

    try {
        const processed = [];
        for (const file of selected) {
            processed.push(await toAttachment(file, source));
        }
        attachments = attachments.concat(processed);
        renderAttachments();

        if (files.length > selected.length) {
            setStatus('error', `Chi giu toi da ${MAX_ATTACHMENTS} tep dinh kem`);
        } else {
            checkHealth();
        }
    } finally {
        isPreparingAttachments = false;
        setComposerState();
    }
}

async function toAttachment(file, source) {
    const relativePath = file.webkitRelativePath || file.name;
    const attachment = {
        id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
        name: file.name,
        path: relativePath,
        size: file.size,
        type: file.type || 'application/octet-stream',
        source,
        kind: detectKind(file, source),
        doclingEligible: false,
        encoding: '',
        data: '',
    };

    if (attachment.kind === 'image') {
        const dimensions = await readImageDimensions(file);
        attachment.previewUrl = URL.createObjectURL(file);
        attachment.width = dimensions.width;
        attachment.height = dimensions.height;
        return attachment;
    }

    if (isTextLike(file)) {
        const rawText = await file.text();
        attachment.content = rawText.slice(0, MAX_TEXT_FILE_BYTES);
        attachment.truncated = rawText.length > MAX_TEXT_FILE_BYTES;
        attachment.kind = 'text';
        return attachment;
    }

    if (supportsDocling(file)) {
        attachment.doclingEligible = true;
        if (file.size <= MAX_BINARY_ATTACHMENT_BYTES) {
            attachment.encoding = 'base64';
            attachment.data = await fileToBase64(file);
            setStatus('loading', `Docling dang xu ly ${attachment.name}...`);
            const processed = await processDoclingAttachment(attachment);
            attachment.content = processed.content || '';
            attachment.truncated = Boolean(processed.truncated);
        }
        return attachment;
    }

    attachment.kind = 'binary';
    return attachment;
}

function detectKind(file, source) {
    if (source === 'image' || file.type.startsWith('image/')) return 'image';
    if (isTextLike(file)) return 'text';
    return 'binary';
}

function isTextLike(file) {
    const name = file.name.toLowerCase();
    if ([
        '.doc', '.docx', '.docm', '.rtf', '.odt',
        '.pdf',
        '.ppt', '.pptx', '.pptm', '.odp',
        '.xls', '.xlsx', '.xlsm', '.xlsb', '.ods'
    ].some((ext) => name.endsWith(ext))) {
        return false;
    }
    if (file.type.startsWith('text/')) return true;
    if (file.type.includes('json') || file.type.includes('javascript') || file.type.includes('xml')) return true;
    return [
        '.md', '.txt', '.py', '.js', '.ts', '.tsx', '.jsx', '.html', '.css', '.scss',
        '.java', '.cs', '.cpp', '.c', '.h', '.hpp', '.go', '.rs', '.php', '.rb',
        '.env', '.toml', '.yaml', '.yml', '.ini', '.cfg', '.sql', '.sh', '.bat',
        '.ps1', '.json', '.xml', '.csv', '.log'
    ].some((ext) => name.endsWith(ext));
}

function supportsDocling(file) {
    const name = file.name.toLowerCase();
    return DOCLING_EXTENSIONS.some((ext) => name.endsWith(ext));
}

function fileToBase64(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => {
            const result = String(reader.result || '');
            const [, base64 = ''] = result.split(',', 2);
            resolve(base64);
        };
        reader.onerror = () => reject(reader.error || new Error('Khong doc duoc file'));
        reader.readAsDataURL(file);
    });
}

async function processDoclingAttachment(attachment) {
    const response = await fetch(`${API}/attachments/process`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(serializeAttachmentForRequest(attachment)),
    });

    if (!response.ok) {
        throw new Error(`Khong xu ly duoc tep (${response.status})`);
    }

    return response.json();
}

function readImageDimensions(file) {
    return new Promise((resolve) => {
        const url = URL.createObjectURL(file);
        const img = new Image();
        img.onload = () => {
            resolve({ width: img.naturalWidth, height: img.naturalHeight });
            URL.revokeObjectURL(url);
        };
        img.onerror = () => {
            resolve({ width: 0, height: 0 });
            URL.revokeObjectURL(url);
        };
        img.src = url;
    });
}

function renderAttachments() {
    attachmentTray.innerHTML = '';
    attachmentTray.hidden = attachments.length === 0;

    for (const attachment of attachments) {
        const card = document.createElement('div');
        card.className = 'attachment-card';
        card.innerHTML = `
          <div class="attachment-meta">
            <div class="attachment-title">${escapeHtml(attachment.path)}</div>
            <div class="attachment-subtitle">${escapeHtml(describeAttachment(attachment))}</div>
          </div>
          <button class="attachment-remove" type="button" data-id="${attachment.id}">x</button>
        `;

        if (attachment.kind === 'image' && attachment.previewUrl) {
            const preview = document.createElement('img');
            preview.className = 'attachment-preview';
            preview.src = attachment.previewUrl;
            preview.alt = attachment.name;
            card.prepend(preview);
        }

        attachmentTray.appendChild(card);
    }

    for (const button of attachmentTray.querySelectorAll('.attachment-remove')) {
        button.addEventListener('click', () => removeAttachment(button.dataset.id));
    }
}

function describeAttachment(attachment) {
    if (attachment.kind === 'text') {
        return `${formatSize(attachment.size)} | text${attachment.truncated ? ' | truncated' : ''}`;
    }
    if (attachment.kind === 'image') {
        const size = formatSize(attachment.size);
        const dims = attachment.width && attachment.height ? ` | ${attachment.width}x${attachment.height}` : '';
        return `${size} | image${dims}`;
    }
    if (attachment.doclingEligible) {
        const note = attachment.content ? ' | docling san sang' : ' | qua lon de parse';
        return `${formatSize(attachment.size)} | binary${note}`;
    }
    return `${formatSize(attachment.size)} | binary`;
}

function removeAttachment(id) {
    const attachment = attachments.find((item) => item.id === id);
    if (attachment?.previewUrl) {
        URL.revokeObjectURL(attachment.previewUrl);
    }
    attachments = attachments.filter((item) => item.id !== id);
    renderAttachments();
}

function clearAttachments() {
    for (const attachment of attachments) {
        if (attachment.previewUrl) {
            URL.revokeObjectURL(attachment.previewUrl);
        }
    }
    attachments = [];
    renderAttachments();
}

async function sendMessage() {
    const text = inputEl.value.trim();
    if ((!text && attachments.length === 0) || isLoading || isPreparingAttachments) return;

    messagesEl.querySelector('.welcome')?.remove();

    const outgoingAttachments = attachments.map(stripAttachmentForHistory);
    const requestAttachments = attachments.map(serializeAttachmentForRequest);
    const payloadText = buildUserMessage(text, outgoingAttachments);

    appendMessage('user', text || '(Khong co text, chi gui dinh kem)', outgoingAttachments);
    history.push({ role: 'user', content: payloadText });

    inputEl.value = '';
    inputEl.style.height = 'auto';
    clearAttachments();
    setLoading(true);

    const typingId = appendTyping();

    try {
        const temperature = parseFloat((tempSlider.value / 10).toFixed(1));

        const response = await fetch(`${API}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                messages: history,
                attachments: requestAttachments,
                temperature,
                thinking: isThinking,
                max_tokens: 8192,
                stream: true,
            }),
        });

        if (!response.ok || !response.body) {
            throw new Error(`HTTP ${response.status}`);
        }

        removeTyping(typingId);
        const aiBubble = appendMessage('ai', '');
        const bubbleEl = aiBubble.querySelector('.bubble-text');

        let fullContent = '';
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            buffer += decoder.decode(value || new Uint8Array(), { stream: !done });

            const events = buffer.split('\n\n');
            buffer = events.pop() || '';

            for (const event of events) {
                const lines = event.split('\n');
                for (const line of lines) {
                    if (!line.startsWith('data: ')) continue;
                    try {
                        const chunk = JSON.parse(line.slice(6));
                        if (chunk.token) {
                            fullContent += chunk.token;
                            bubbleEl.textContent = fullContent;
                            scrollToBottom();
                        }
                    } catch {
                        // Wait for the next complete SSE event.
                    }
                }
            }

            if (done) {
                if (buffer.trim()) {
                    const lines = buffer.split('\n');
                    for (const line of lines) {
                        if (!line.startsWith('data: ')) continue;
                        try {
                            const chunk = JSON.parse(line.slice(6));
                            if (chunk.token) fullContent += chunk.token;
                        } catch {
                            // Ignore trailing partial data.
                        }
                    }
                }
                bubbleEl.textContent = fullContent;
                scrollToBottom();
                break;
            }
        }

        history.push({ role: 'assistant', content: fullContent });
    } catch (err) {
        removeTyping(typingId);
        appendMessage('ai', `Loi: ${err.message}\n\nHay chac chan backend dang chay (start.bat).`);
    }

    setLoading(false);
}

async function summarizeAttachments() {
    if (!attachments.length || isLoading || isPreparingAttachments) return;

    messagesEl.querySelector('.welcome')?.remove();

    const outgoingAttachments = attachments.map(stripAttachmentForHistory);
    const requestAttachments = attachments.map(serializeAttachmentForSummaryRequest);

    appendMessage('user', 'Hay chay pipeline AI summary cho cac tep dinh kem.', outgoingAttachments);
    clearAttachments();
    setLoading(true);

    const typingId = appendTyping();

    try {
        const response = await fetch(`${API}/summaries/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                attachments: requestAttachments,
                temperature: 0.2,
                cluster_max_tokens: 1024,
                final_max_tokens: 1800,
            }),
        });

        const payload = await response.json();
        if (!response.ok || payload.error) {
            throw new Error(payload.error || `HTTP ${response.status}`);
        }

        removeTyping(typingId);
        appendMessage('ai', buildSummaryMessage(payload));
    } catch (err) {
        removeTyping(typingId);
        appendMessage('ai', `Loi khi tao AI summary: ${err.message}`);
    }

    setLoading(false);
}

function buildUserMessage(text, outgoingAttachments) {
    const parts = [];
    parts.push(text || 'Nguoi dung khong nhap them text, chi gui tep dinh kem.');

    if (outgoingAttachments.length) {
        parts.push('Thong tin dinh kem:');

        for (const attachment of outgoingAttachments) {
            if (attachment.kind === 'text') {
                parts.push(
                    [
                        `FILE: ${attachment.path}`,
                        `SIZE: ${formatSize(attachment.size)}`,
                        attachment.truncated ? 'NOTE: Noi dung da bi cat bot vi qua dai.' : '',
                        'CONTENT:',
                        attachment.content || '(Rong)',
                    ].filter(Boolean).join('\n')
                );
                continue;
            }

            if (attachment.kind === 'image') {
                parts.push(
                    [
                        `IMAGE: ${attachment.path}`,
                        `SIZE: ${formatSize(attachment.size)}`,
                        attachment.width && attachment.height ? `DIMENSIONS: ${attachment.width}x${attachment.height}` : '',
                        'NOTE: Day la model text-only, anh duoc gui duoi dang metadata chua co phan tich pixel.',
                    ].filter(Boolean).join('\n')
                );
                continue;
            }

            parts.push(
                [
                    `FILE: ${attachment.path}`,
                    `SIZE: ${formatSize(attachment.size)}`,
                    attachment.doclingEligible
                        ? attachment.content
                            ? 'NOTE: Docling da trich xuat xong noi dung tep nay truoc khi gui.'
                            : `NOTE: Tep ho tro Docling nhung vuot gioi han ${formatSize(MAX_BINARY_ATTACHMENT_BYTES)} nen chi gui metadata.`
                        : 'NOTE: Tep nhi phan hoac dinh dang office chi gui metadata, khong gui noi dung vao model.',
                ].join('\n')
            );
        }
    }

    return parts.join('\n\n');
}

function stripAttachmentForHistory(attachment) {
    return {
        kind: attachment.kind,
        name: attachment.name,
        path: attachment.path,
        size: attachment.size,
        width: attachment.width || 0,
        height: attachment.height || 0,
        truncated: Boolean(attachment.truncated),
        content: attachment.content || '',
        doclingEligible: Boolean(attachment.doclingEligible),
        encoding: attachment.content ? 'prepared' : attachment.encoding || '',
    };
}

function serializeAttachmentForRequest(attachment) {
    return {
        kind: attachment.kind,
        name: attachment.name,
        path: attachment.path,
        size: attachment.size,
        mime_type: attachment.type || '',
        truncated: Boolean(attachment.truncated),
        content: attachment.content || '',
        encoding: attachment.content ? 'prepared' : attachment.encoding || '',
        data: attachment.content ? '' : attachment.data || '',
    };
}

function serializeAttachmentForSummaryRequest(attachment) {
    return {
        kind: attachment.kind,
        name: attachment.name,
        path: attachment.path,
        size: attachment.size,
        mime_type: attachment.type || '',
        truncated: Boolean(attachment.truncated),
        content: attachment.content || '',
        encoding: attachment.encoding || '',
        data: attachment.data || '',
    };
}

function buildSummaryMessage(payload) {
    const parts = [];
    parts.push('Tom tat tong the');
    parts.push(payload.overall_summary || '(Khong co noi dung tom tat tong the)');

    if (payload.artifacts?.run_dir) {
        parts.push(`Output folder\n${payload.artifacts.run_dir}`);
    }

    if (Array.isArray(payload.clusters) && payload.clusters.length) {
        parts.push('Cum chu de');
        for (const cluster of payload.clusters) {
            const header = `- ${cluster.title || cluster.cluster_id} (${cluster.size} chunk)`;
            const previews = (cluster.representative_previews || []).slice(0, 3).map((item) => `  evidence: ${item}`);
            parts.push([header, cluster.summary || '', ...previews].filter(Boolean).join('\n'));
        }
    }

    if (payload.debug) {
        parts.push(
            `Thong ke: ${payload.debug.document_count} tai lieu, ${payload.debug.chunk_count} chunk, ${payload.debug.cluster_count} cluster`
        );
    }

    return parts.join('\n\n');
}

function appendMessage(role, content, messageAttachments = []) {
    const div = document.createElement('div');
    div.className = `msg ${role}`;
    div.innerHTML = `
    <div class="avatar">${role === 'user' ? 'U' : 'AI'}</div>
    <div class="bubble">
      <div class="bubble-text"></div>
    </div>`;

    div.querySelector('.bubble-text').textContent = content;

    if (messageAttachments.length) {
        const bubble = div.querySelector('.bubble');
        const list = document.createElement('div');
        list.className = 'message-attachments';

        for (const attachment of messageAttachments) {
            const item = document.createElement('div');
            item.className = 'message-attachment';
            item.textContent = attachment.path;
            list.appendChild(item);
        }

        bubble.appendChild(list);
    }

    messagesEl.appendChild(div);
    scrollToBottom();
    return div;
}

function appendTyping() {
    const id = 'typing-' + Date.now();
    const div = document.createElement('div');
    div.id = id;
    div.className = 'msg ai typing';
    div.innerHTML = `
    <div class="avatar">AI</div>
    <div class="bubble">
      <div class="dot-pulse"></div>
      <div class="dot-pulse"></div>
      <div class="dot-pulse"></div>
    </div>`;
    messagesEl.appendChild(div);
    scrollToBottom();
    return id;
}

function removeTyping(id) {
    document.getElementById(id)?.remove();
}

function setLoading(val) {
    isLoading = val;
    setComposerState();
}

function setComposerState() {
    const disabled = isLoading || isPreparingAttachments;
    sendBtn.disabled = disabled;
    inputEl.disabled = disabled;
    if (summaryBtn) {
        summaryBtn.disabled = disabled || attachments.length === 0;
    }
    for (const button of composerAttachButtons) {
        button.disabled = disabled;
    }
}

function scrollToBottom() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
}

function formatSize(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function escapeHtml(text) {
    return String(text)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}

window.openPicker = openPicker;
window.suggest = suggest;
window.sendMessage = sendMessage;
window.summarizeAttachments = summarizeAttachments;
