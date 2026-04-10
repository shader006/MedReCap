const API = "http://localhost:8888";
const MAX_ATTACHMENTS = 24;
const MAX_TEXT_FILE_BYTES = 16000;
const MAX_BINARY_ATTACHMENT_BYTES = 25 * 1024 * 1024;
const DOCLING_EXTENSIONS = [
  ".pdf", ".docx", ".doc", ".docm", ".rtf", ".odt",
  ".pptx", ".ppt", ".pptm", ".odp",
  ".xlsx", ".xls", ".xlsm", ".xlsb", ".ods",
  ".html", ".htm", ".md", ".csv", ".xml"
];
const THEME_KEY = "nuist-summary-theme";

const PIPELINE_STAGES = [
  { key: "collect", title: "Chuẩn bị tài liệu", note: "Đọc input đã chọn và chuẩn hóa attachment." },
  { key: "extract", title: "Trích xuất nội dung", note: "Chuyển tài liệu thành văn bản để pipeline có thể xử lý." },
  { key: "cluster", title: "Phân cụm chủ đề", note: "Chunk tài liệu, tạo embedding và gom các ý liên quan." },
  { key: "summarize", title: "Sinh tóm tắt", note: "Viết cluster summaries rồi tổng hợp thành final summary." },
  { key: "finalize", title: "Hoàn tất output", note: "Lưu artifact, final summary và thống kê của lần chạy." }
];

let attachments = [];
let isLoading = false;
let phaseTimer = null;
let activeStageIndex = -1;
let lastPayload = null;
let lastSummaryMarkdown = "";

const summaryPickerBtn = document.getElementById("summaryPickerBtn");
const pickerMenu = document.getElementById("pickerMenu");
const pickFileBtn = document.getElementById("pickFileBtn");
const pickFolderBtn = document.getElementById("pickFolderBtn");
const runBtn = document.getElementById("runBtn");
const clearBtn = document.getElementById("clearBtn");
const copyBtn = document.getElementById("copyBtn");
const downloadBtn = document.getElementById("downloadBtn");
const themeToggleBtn = document.getElementById("themeToggleBtn");
const fileInput = document.getElementById("fileInput");
const folderInput = document.getElementById("folderInput");
const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");
const modelStatus = document.getElementById("modelStatus");
const selectionEmpty = document.getElementById("selectionEmpty");
const selectionList = document.getElementById("selectionList");
const selectionCountPill = document.getElementById("selectionCountPill");
const selectionSizePill = document.getElementById("selectionSizePill");
const runDirValue = document.getElementById("runDirValue");
const finalSummaryPath = document.getElementById("finalSummaryPath");
const documentsValue = document.getElementById("documentsValue");
const clustersValue = document.getElementById("clustersValue");
const summaryOutput = document.getElementById("summaryOutput");
const clusterList = document.getElementById("clusterList");
const artifactList = document.getElementById("artifactList");
const stageList = document.getElementById("stageList");
const runBadge = document.getElementById("runBadge");

init();

function init() {
  initializeTheme();
  renderStages();
  bindEvents();
  updateSelectionSummary();
  setRunState("idle");
  checkHealth();
  fetchGpuStats();
  setInterval(checkHealth, 30000);
  setInterval(fetchGpuStats, 3000);
}

function bindEvents() {
  summaryPickerBtn.addEventListener("click", () => {
    pickerMenu.hidden = !pickerMenu.hidden;
  });

  document.addEventListener("click", (event) => {
    if (!pickerMenu.hidden && !pickerMenu.contains(event.target) && event.target !== summaryPickerBtn) {
      pickerMenu.hidden = true;
    }
  });

  pickFileBtn.addEventListener("click", () => {
    pickerMenu.hidden = true;
    fileInput.click();
  });

  pickFolderBtn.addEventListener("click", () => {
    pickerMenu.hidden = true;
    folderInput.click();
  });

  fileInput.addEventListener("change", (event) => handleSelectedFiles(event.target.files, "file", event.target));
  folderInput.addEventListener("change", (event) => handleSelectedFiles(event.target.files, "folder", event.target));

  runBtn.addEventListener("click", runSummaryPipeline);
  clearBtn.addEventListener("click", clearAttachments);
  copyBtn.addEventListener("click", copySummaryToClipboard);
  downloadBtn.addEventListener("click", downloadSummaryMarkdown);
  themeToggleBtn.addEventListener("click", toggleTheme);
}

async function checkHealth() {
  try {
    const response = await fetch(`${API}/health`);
    const payload = await response.json();
    const modelText = payload.model_ready
      ? `${payload.model} | ctx ${payload.summary_ctx}`
      : "Model chưa sẵn sàng";
    setStatus(payload.model_ready ? "online" : "loading", payload.model_ready ? "Backend sẵn sàng" : "Đang tải model...");
    modelStatus.textContent = modelText;
  } catch {
    setStatus("error", "Không kết nối được backend");
    modelStatus.textContent = "Hãy kiểm tra server ở cổng 8888.";
  }
}

async function fetchGpuStats() {
  try {
    const response = await fetch(`${API}/gpu`);
    const payload = await response.json();
    if (!payload.available || !Array.isArray(payload.gpus) || payload.gpus.length === 0) {
      resetGpuStats();
      return;
    }

    const gpu = payload.gpus[0];
    document.getElementById("gpuSection").hidden = false;
    document.getElementById("gpuName").textContent = gpu.name;
    document.getElementById("gpuUtil").textContent = `${gpu.utilization}%`;
    document.getElementById("gpuUtilBar").style.width = `${gpu.utilization}%`;

    const usedGB = (gpu.memory_used / 1024).toFixed(1);
    const totalGB = (gpu.memory_total / 1024).toFixed(1);
    document.getElementById("gpuMem").textContent = `${usedGB}/${totalGB} GB`;
    document.getElementById("gpuMemBar").style.width = `${gpu.memory_percent}%`;

    const tempRow = document.getElementById("gpuTempRow");
    document.getElementById("gpuTemp").textContent = `${gpu.temperature} C`;
    tempRow.className = "gpu-row temp-row";
    if (gpu.temperature >= 85) {
      tempRow.classList.add("danger");
    } else if (gpu.temperature >= 70) {
      tempRow.classList.add("hot");
    }
  } catch {
    resetGpuStats();
  }
}

function resetGpuStats() {
  document.getElementById("gpuSection").hidden = true;
  document.getElementById("gpuName").textContent = "-";
  document.getElementById("gpuUtil").textContent = "-%";
  document.getElementById("gpuUtilBar").style.width = "0%";
  document.getElementById("gpuMem").textContent = "-";
  document.getElementById("gpuMemBar").style.width = "0%";
  document.getElementById("gpuTemp").textContent = "- C";
  document.getElementById("gpuTempRow").className = "gpu-row temp-row";
}

function setStatus(type, text) {
  statusDot.className = `status-dot ${type}`;
  statusText.textContent = text;
}

function initializeTheme() {
  const savedTheme = localStorage.getItem(THEME_KEY);
  const preferredDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  const theme = savedTheme || (preferredDark ? "dark" : "light");
  applyTheme(theme);
}

function toggleTheme() {
  const currentTheme = document.documentElement.dataset.theme === "dark" ? "dark" : "light";
  const nextTheme = currentTheme === "dark" ? "light" : "dark";
  applyTheme(nextTheme);
  localStorage.setItem(THEME_KEY, nextTheme);
}

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  const isDark = theme === "dark";
  themeToggleBtn.textContent = isDark ? "Light mode" : "Dark mode";
  themeToggleBtn.setAttribute("aria-pressed", String(isDark));
}

async function handleSelectedFiles(fileList, source, input) {
  const files = Array.from(fileList || []);
  input.value = "";

  if (!files.length) {
    return;
  }

  const availableSlots = Math.max(0, MAX_ATTACHMENTS - attachments.length);
  const selected = files.slice(0, availableSlots);
  const ignoredCount = files.length - selected.length;

  if (!selected.length) {
    setStatus("error", `Đã đạt giới hạn ${MAX_ATTACHMENTS} tài liệu.`);
    return;
  }

  setStatus("loading", "Đang chuẩn bị tài liệu...");

  for (const file of selected) {
    const prepared = await toAttachment(file, source);
    attachments.push(prepared);
  }

  renderSelectionList();
  updateSelectionSummary();
  setActionState();

  if (ignoredCount > 0) {
    setStatus("error", `Chỉ giữ tối đa ${MAX_ATTACHMENTS} tài liệu trong một lần chạy.`);
  } else {
    checkHealth();
  }
}

async function toAttachment(file, source) {
  const path = file.webkitRelativePath || file.name;
  const attachment = {
    id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
    name: file.name,
    path,
    size: file.size,
    type: file.type || "application/octet-stream",
    source,
    kind: "binary",
    content: "",
    truncated: false,
    encoding: "",
    data: "",
    doclingEligible: supportsDocling(file),
    unsupported: false
  };

  if (file.type.startsWith("image/")) {
    attachment.unsupported = true;
    return attachment;
  }

  if (isTextLike(file)) {
    attachment.kind = "text";
    const rawText = await file.text();
    attachment.content = rawText.slice(0, MAX_TEXT_FILE_BYTES);
    attachment.truncated = rawText.length > MAX_TEXT_FILE_BYTES;
    return attachment;
  }

  if (attachment.doclingEligible && file.size <= MAX_BINARY_ATTACHMENT_BYTES) {
    attachment.encoding = "base64";
    attachment.data = await fileToBase64(file);
    return attachment;
  }

  if (attachment.doclingEligible && file.size > MAX_BINARY_ATTACHMENT_BYTES) {
    attachment.truncated = true;
    return attachment;
  }

  attachment.unsupported = true;
  return attachment;
}

function renderSelectionList() {
  if (!attachments.length) {
    selectionEmpty.hidden = false;
    selectionList.hidden = true;
    selectionList.innerHTML = "";
    return;
  }

  selectionEmpty.hidden = true;
  selectionList.hidden = false;
  selectionList.innerHTML = attachments.map((attachment) => {
    const tag = attachment.unsupported ? "Unsupported" : attachment.kind === "text" ? "Text" : attachment.doclingEligible ? "Docling" : "Binary";
    return `
      <article class="selection-item">
        <div class="selection-top">
          <div>
            <div class="selection-path">${escapeHtml(attachment.path)}</div>
            <div class="selection-subtitle">${escapeHtml(describeAttachment(attachment))}</div>
          </div>
          <div class="selection-tag">${tag}</div>
        </div>
      </article>
    `;
  }).join("");
}

function updateSelectionSummary() {
  const supportedCount = attachments.filter((item) => !item.unsupported).length;
  const totalSize = attachments.reduce((sum, item) => sum + item.size, 0);
  selectionCountPill.textContent = attachments.length
    ? `${supportedCount}/${attachments.length} tài liệu hợp lệ`
    : "0 tài liệu";
  selectionSizePill.textContent = formatSize(totalSize);
  setActionState();
}

function setActionState() {
  const hasSupported = attachments.some((item) => !item.unsupported);
  runBtn.disabled = isLoading || !hasSupported;
  clearBtn.disabled = isLoading || attachments.length === 0;
  summaryPickerBtn.disabled = isLoading;
  copyBtn.disabled = !lastSummaryMarkdown;
  downloadBtn.disabled = !lastSummaryMarkdown;
}

async function runSummaryPipeline() {
  const requestAttachments = attachments.filter((item) => !item.unsupported).map(serializeAttachmentForSummaryRequest);
  if (!requestAttachments.length || isLoading) {
    return;
  }

  isLoading = true;
  lastPayload = null;
  lastSummaryMarkdown = "";
  setActionState();
  setRunState("loading");
  setStatus("loading", "Pipeline đang chạy...");
  runDirValue.textContent = "-";
  finalSummaryPath.textContent = "-";
  documentsValue.textContent = "0";
  clustersValue.textContent = "0";
  setSummaryState("Đang chạy pipeline, vui lòng đợi trong giây lát.");
  setArtifacts([]);
  setClusters([]);
  startStageAnimation();

  try {
    const response = await fetch(`${API}/summaries/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        attachments: requestAttachments,
        temperature: 0.2,
        cluster_max_tokens: 1024,
        final_max_tokens: 1800
      })
    });

    const payload = await response.json();
    if (!response.ok || payload.error) {
      throw new Error(payload.error || `HTTP ${response.status}`);
    }

    lastPayload = payload;
    stopStageAnimation("done");
    setRunState("done");
    setStatus("online", "Pipeline hoàn tất");
    hydrateDashboard(payload);
  } catch (error) {
    stopStageAnimation("error");
    setRunState("error");
    setStatus("error", "Pipeline thất bại");
    setSummaryError(`Không thể tạo final summary: ${error.message}`);
  } finally {
    isLoading = false;
    setActionState();
  }
}

function hydrateDashboard(payload) {
  const artifacts = payload.artifacts || {};
  const debug = payload.debug || {};
  const clusters = Array.isArray(payload.clusters) ? payload.clusters : [];

  runDirValue.textContent = artifacts.run_dir || "-";
  finalSummaryPath.textContent = artifacts.final_summary || "-";
  documentsValue.textContent = String(debug.document_count || 0);
  clustersValue.textContent = String(clusters.length);

  setArtifacts(Object.entries(artifacts));
  setClusters(clusters);

  lastSummaryMarkdown = buildExportMarkdown(payload);
  renderMarkdownToSummary(payload.overall_summary || "(Không có nội dung tóm tắt)");
}

function setArtifacts(entries) {
  if (!entries.length) {
    artifactList.innerHTML = `<div class="artifact-empty">Chưa có artifact nào.</div>`;
    return;
  }

  artifactList.innerHTML = entries.map(([label, value]) => `
    <article class="artifact-item">
      <div class="artifact-label">${humanizeArtifactKey(label)}</div>
      <div class="artifact-path">${escapeHtml(String(value || "-"))}</div>
    </article>
  `).join("");
}

function setClusters(clusters) {
  if (!clusters.length) {
    clusterList.innerHTML = `<div class="cluster-empty">Khi pipeline hoàn tất, các cụm nội dung sẽ xuất hiện ở đây.</div>`;
    return;
  }

  clusterList.innerHTML = clusters.map((cluster) => `
    <article class="cluster-card">
      <div class="cluster-top">
        <div class="cluster-title">${escapeHtml(cluster.title || cluster.cluster_id || "Untitled cluster")}</div>
        <div class="cluster-size">${cluster.size || 0} chunk</div>
      </div>
      <div class="cluster-summary">${escapeHtml(cluster.summary || "")}</div>
      ${(cluster.representative_previews || []).length ? `
        <div class="cluster-evidence">
          ${(cluster.representative_previews || []).slice(0, 3).map((item) => `
            <div class="cluster-evidence-item">${escapeHtml(item)}</div>
          `).join("")}
        </div>
      ` : ""}
    </article>
  `).join("");
}

function renderStages() {
  stageList.innerHTML = PIPELINE_STAGES.map((stage, index) => `
    <article class="stage-item pending" data-stage="${stage.key}">
      <div class="stage-index">${index + 1}</div>
      <div>
        <div class="stage-title">${stage.title}</div>
        <div class="stage-note">${stage.note}</div>
      </div>
      <div class="stage-state">Pending</div>
    </article>
  `).join("");
}

function startStageAnimation() {
  activeStageIndex = 0;
  syncStageUI("active");
  clearInterval(phaseTimer);
  phaseTimer = window.setInterval(() => {
    activeStageIndex = Math.min(activeStageIndex + 1, PIPELINE_STAGES.length - 1);
    syncStageUI("active");
  }, 1800);
}

function stopStageAnimation(result) {
  clearInterval(phaseTimer);
  phaseTimer = null;
  syncStageUI(result);
}

function syncStageUI(result) {
  const items = Array.from(stageList.querySelectorAll(".stage-item"));
  items.forEach((item, index) => {
    const stateEl = item.querySelector(".stage-state");
    item.className = "stage-item";

    if (result === "error") {
      if (index < activeStageIndex) {
        item.classList.add("done");
        stateEl.textContent = "Done";
      } else if (index === activeStageIndex) {
        item.classList.add("error");
        stateEl.textContent = "Error";
      } else {
        item.classList.add("pending");
        stateEl.textContent = "Pending";
      }
      return;
    }

    if (result === "done") {
      item.classList.add("done");
      stateEl.textContent = "Done";
      return;
    }

    if (index < activeStageIndex) {
      item.classList.add("done");
      stateEl.textContent = "Done";
    } else if (index === activeStageIndex) {
      item.classList.add("active");
      stateEl.textContent = "Running";
    } else {
      item.classList.add("pending");
      stateEl.textContent = "Pending";
    }
  });
}

function setRunState(state) {
  runBadge.className = "run-badge";
  if (state === "loading") {
    runBadge.classList.add("loading");
    runBadge.textContent = "Running";
  } else if (state === "error") {
    runBadge.classList.add("error");
    runBadge.textContent = "Error";
  } else if (state === "done") {
    runBadge.textContent = "Completed";
  } else {
    runBadge.textContent = "Idle";
  }
}

function setSummaryState(text) {
  summaryOutput.innerHTML = `<div class="summary-state">${escapeHtml(text)}</div>`;
}

function setSummaryError(text) {
  summaryOutput.innerHTML = `<div class="summary-state summary-error">${escapeHtml(text)}</div>`;
}

function renderMarkdownToSummary(markdown) {
  summaryOutput.innerHTML = `<article class="summary-markdown">${renderMarkdown(markdown)}</article>`;
}

function clearAttachments() {
  attachments = [];
  renderSelectionList();
  updateSelectionSummary();
  setStatus("online", "Đã xóa lựa chọn");
}

async function copySummaryToClipboard() {
  if (!lastSummaryMarkdown) {
    return;
  }

  try {
    await navigator.clipboard.writeText(lastSummaryMarkdown);
    setStatus("online", "Đã copy final summary");
  } catch {
    setStatus("error", "Không thể copy vào clipboard");
  }
}

function downloadSummaryMarkdown() {
  if (!lastSummaryMarkdown) {
    return;
  }

  const blob = new Blob([lastSummaryMarkdown], { type: "text/markdown;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = "final-summary.md";
  anchor.click();
  URL.revokeObjectURL(url);
}

function buildExportMarkdown(payload) {
  const parts = [];
  const artifacts = payload.artifacts || {};
  const debug = payload.debug || {};
  const clusters = Array.isArray(payload.clusters) ? payload.clusters : [];

  parts.push("# Final Summary");
  parts.push("");
  parts.push(payload.overall_summary || "(Không có nội dung)");

  parts.push("");
  parts.push("## Run Info");
  parts.push(`- Model: ${payload.model || "-"}`);
  parts.push(`- Run folder: ${artifacts.run_dir || "-"}`);
  parts.push(`- Final summary file: ${artifacts.final_summary || "-"}`);
  parts.push(`- Documents: ${debug.document_count || 0}`);
  parts.push(`- Clusters: ${clusters.length}`);

  if (clusters.length) {
    parts.push("");
    parts.push("## Topic Clusters");
    for (const cluster of clusters) {
      parts.push("");
      parts.push(`### ${cluster.title || cluster.cluster_id || "Untitled cluster"}`);
      parts.push(cluster.summary || "(Không có nội dung)");
      if (Array.isArray(cluster.representative_previews) && cluster.representative_previews.length) {
        parts.push("");
        for (const preview of cluster.representative_previews.slice(0, 3)) {
          parts.push(`- ${preview}`);
        }
      }
    }
  }

  return parts.join("\n");
}

function serializeAttachmentForSummaryRequest(attachment) {
  return {
    kind: attachment.kind,
    name: attachment.name,
    path: attachment.path,
    size: attachment.size,
    mime_type: attachment.type || "",
    truncated: Boolean(attachment.truncated),
    content: attachment.content || "",
    encoding: attachment.encoding || "",
    data: attachment.data || ""
  };
}

function describeAttachment(attachment) {
  if (attachment.unsupported) {
    return `${formatSize(attachment.size)} | chưa hỗ trợ trong viewer này`;
  }
  if (attachment.kind === "text") {
    return `${formatSize(attachment.size)} | text${attachment.truncated ? " | đã cắt bớt" : ""}`;
  }
  if (attachment.doclingEligible) {
    return `${formatSize(attachment.size)} | file parse qua Docling${attachment.truncated ? " | quá lớn" : ""}`;
  }
  return `${formatSize(attachment.size)} | binary`;
}

function isTextLike(file) {
  const name = file.name.toLowerCase();
  if ([
    ".doc", ".docx", ".docm", ".rtf", ".odt",
    ".pdf",
    ".ppt", ".pptx", ".pptm", ".odp",
    ".xls", ".xlsx", ".xlsm", ".xlsb", ".ods"
  ].some((ext) => name.endsWith(ext))) {
    return false;
  }
  if (file.type.startsWith("text/")) return true;
  if (file.type.includes("json") || file.type.includes("javascript") || file.type.includes("xml")) return true;
  return [
    ".md", ".txt", ".py", ".js", ".ts", ".tsx", ".jsx", ".html", ".css", ".scss",
    ".java", ".cs", ".cpp", ".c", ".h", ".hpp", ".go", ".rs", ".php", ".rb",
    ".env", ".toml", ".yaml", ".yml", ".ini", ".cfg", ".sql", ".sh", ".bat",
    ".ps1", ".json", ".xml", ".csv", ".log"
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
      const result = String(reader.result || "");
      const [, base64 = ""] = result.split(",", 2);
      resolve(base64);
    };
    reader.onerror = () => reject(reader.error || new Error("Không thể đọc file"));
    reader.readAsDataURL(file);
  });
}

function formatSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function humanizeArtifactKey(key) {
  return String(key)
    .replace(/_/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function escapeHtml(text) {
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function renderMarkdown(markdown) {
  const normalized = String(markdown || "").replace(/\r\n/g, "\n");
  const fencePattern = /```([\w-]*)\n([\s\S]*?)```/g;
  const segments = [];
  let lastIndex = 0;
  let match;

  while ((match = fencePattern.exec(normalized)) !== null) {
    if (match.index > lastIndex) {
      segments.push({ type: "text", value: normalized.slice(lastIndex, match.index) });
    }
    segments.push({ type: "code", lang: match[1], value: match[2] });
    lastIndex = match.index + match[0].length;
  }

  if (lastIndex < normalized.length) {
    segments.push({ type: "text", value: normalized.slice(lastIndex) });
  }

  return segments.map((segment) => {
    if (segment.type === "code") {
      return `<pre><code>${escapeHtml(segment.value)}</code></pre>`;
    }
    return renderMarkdownText(segment.value);
  }).join("");
}

function renderMarkdownText(text) {
  const blocks = text.split(/\n\s*\n/).map((block) => block.trim()).filter(Boolean);
  return blocks.map((block) => {
    if (/^###\s+/.test(block)) {
      return `<h3>${renderInlineMarkdown(block.replace(/^###\s+/, ""))}</h3>`;
    }
    if (/^##\s+/.test(block)) {
      return `<h2>${renderInlineMarkdown(block.replace(/^##\s+/, ""))}</h2>`;
    }
    if (/^#\s+/.test(block)) {
      return `<h1>${renderInlineMarkdown(block.replace(/^#\s+/, ""))}</h1>`;
    }
    if (/^>\s+/.test(block)) {
      const lines = block.split("\n").map((line) => line.replace(/^>\s?/, ""));
      return `<blockquote>${lines.map(renderInlineMarkdown).join("<br>")}</blockquote>`;
    }
    if (/^(-|\*)\s+/m.test(block) && block.split("\n").every((line) => /^(-|\*)\s+/.test(line))) {
      const items = block.split("\n").map((line) => `<li>${renderInlineMarkdown(line.replace(/^(-|\*)\s+/, ""))}</li>`).join("");
      return `<ul>${items}</ul>`;
    }
    if (/^\d+\.\s+/m.test(block) && block.split("\n").every((line) => /^\d+\.\s+/.test(line))) {
      const items = block.split("\n").map((line) => `<li>${renderInlineMarkdown(line.replace(/^\d+\.\s+/, ""))}</li>`).join("");
      return `<ol>${items}</ol>`;
    }
    return `<p>${block.split("\n").map(renderInlineMarkdown).join("<br>")}</p>`;
  }).join("");
}

function renderInlineMarkdown(text) {
  return escapeHtml(text)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\*([^*]+)\*/g, "<em>$1</em>");
}
