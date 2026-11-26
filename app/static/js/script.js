/**
 * AI-Powered Smart Study Assistant
 * Phase 2 - Frontend Interactivity: File Upload, Validation, Progress, and Diagnostics
 */

document.addEventListener("DOMContentLoaded", () => {
    // -------------------------------------------------------------------------
    // 1. Health Diagnostics (from Phase 1)
    // -------------------------------------------------------------------------
    const statusIndicator = document.getElementById("system-status-indicator");
    const statusText = document.getElementById("system-status-text");
    const refreshBtn = document.getElementById("refresh-status-btn");
    
    const diagAppStatus = document.getElementById("diag-app-status");
    const diagFramework = document.getElementById("diag-framework");
    const diagEnv = document.getElementById("diag-env");
    const diagStorage = document.getElementById("diag-storage");

    async function fetchSystemHealth() {
        if (refreshBtn) refreshBtn.classList.add("spinning");

        try {
            const response = await fetch("/health");
            if (!response.ok) throw new Error(`HTTP Error ${response.status}`);

            const data = await response.json();

            if (statusIndicator && statusText) {
                statusText.textContent = "Backend Online";
                statusIndicator.style.background = "rgba(16, 185, 129, 0.12)";
                statusIndicator.style.borderColor = "rgba(16, 185, 129, 0.3)";
            }

            if (diagAppStatus) diagAppStatus.textContent = data.status.toUpperCase();
            if (diagFramework) diagFramework.textContent = `${data.service} v${data.version}`;
            if (diagEnv) diagEnv.textContent = data.environment.toUpperCase();
            if (diagStorage) {
                diagStorage.textContent = data.upload_folder_ready ? "Ready (Active)" : "Offline";
            }

        } catch (error) {
            console.error("Health check error:", error);

            if (statusIndicator && statusText) {
                statusText.textContent = "Backend Offline";
                statusIndicator.style.background = "rgba(239, 68, 68, 0.12)";
                statusIndicator.style.borderColor = "rgba(239, 68, 68, 0.3)";
            }

            if (diagAppStatus) {
                diagAppStatus.textContent = "DISCONNECTED";
                diagAppStatus.className = "diag-value";
                diagAppStatus.style.color = "#ef4444";
            }
        } finally {
            if (refreshBtn) {
                setTimeout(() => refreshBtn.classList.remove("spinning"), 400);
            }
        }
    }

    if (refreshBtn) {
        refreshBtn.addEventListener("click", fetchSystemHealth);
    }
    fetchSystemHealth();

    // -------------------------------------------------------------------------
    // 2. Alert Notification Helper
    // -------------------------------------------------------------------------
    const clientAlert = document.getElementById("client-alert");

    function showAlert(message, type = "error") {
        if (!clientAlert) return;
        clientAlert.classList.remove("hidden");
        const icon = type === "success" ? "✓" : (type === "warning" ? "⚠️" : "✕");
        clientAlert.innerHTML = `
            <div class="alert alert-${type}">
                <span class="alert-icon">${icon}</span>
                <span class="alert-message">${message}</span>
                <button type="button" class="alert-close" onclick="this.parentElement.remove()">×</button>
            </div>
        `;
        clientAlert.scrollIntoView({ behavior: "smooth", block: "center" });
    }

    function clearAlert() {
        if (clientAlert) {
            clientAlert.innerHTML = "";
            clientAlert.classList.add("hidden");
        }
    }

    // -------------------------------------------------------------------------
    // 3. PDF Upload, Drag-and-Drop, Client Validation
    // -------------------------------------------------------------------------
    const uploadForm = document.getElementById("upload-form");
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("pdf-file-input");
    const dropzonePrompt = document.getElementById("dropzone-prompt");
    const selectedPreview = document.getElementById("selected-file-preview");
    const previewFileName = document.getElementById("preview-file-name");
    const previewFileSize = document.getElementById("preview-file-size");
    const btnClearFile = document.getElementById("btn-clear-file");
    const btnSubmitUpload = document.getElementById("btn-submit-upload");
    const progressContainer = document.getElementById("upload-progress-container");
    const progressBarFill = document.getElementById("progress-bar-fill");
    const progressStatusText = document.getElementById("progress-status-text");

    const MAX_FILE_SIZE_BYTES = 16 * 1024 * 1024; // 16 MB

    function formatBytes(bytes) {
        if (bytes === 0) return "0 Bytes";
        const k = 1024;
        const sizes = ["Bytes", "KB", "MB", "GB"];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
    }

    function handleFileSelection(file) {
        clearAlert();

        if (!file) {
            resetFileInput();
            return;
        }

        // Validate Extension
        const fileName = file.name || "";
        if (!fileName.toLowerCase().endsWith(".pdf")) {
            showAlert("Unsupported file format. Only PDF files are supported.", "error");
            resetFileInput();
            return;
        }

        // Validate Size (0 bytes)
        if (file.size === 0) {
            showAlert("The selected PDF file is empty (0 bytes). Please select a valid document.", "error");
            resetFileInput();
            return;
        }

        // Validate Size (Max 16 MB)
        if (file.size > MAX_FILE_SIZE_BYTES) {
            showAlert(`File exceeds maximum upload limit of 16 MB (Selected: ${formatBytes(file.size)}).`, "error");
            resetFileInput();
            return;
        }

        // Display File Preview
        if (previewFileName) previewFileName.textContent = fileName;
        if (previewFileSize) previewFileSize.textContent = formatBytes(file.size);

        if (dropzonePrompt) dropzonePrompt.classList.add("hidden");
        if (selectedPreview) selectedPreview.classList.remove("hidden");
        if (btnSubmitUpload) btnSubmitUpload.disabled = false;
    }

    function resetFileInput() {
        if (fileInput) fileInput.value = "";
        if (dropzonePrompt) dropzonePrompt.classList.remove("hidden");
        if (selectedPreview) selectedPreview.classList.add("hidden");
        if (btnSubmitUpload) btnSubmitUpload.disabled = true;
        if (progressContainer) progressContainer.classList.add("hidden");
    }

    if (fileInput) {
        fileInput.addEventListener("change", (e) => {
            if (e.target.files && e.target.files[0]) {
                handleFileSelection(e.target.files[0]);
            }
        });
    }

    if (btnClearFile) {
        btnClearFile.addEventListener("click", (e) => {
            e.stopPropagation();
            resetFileInput();
        });
    }

    // Drag and drop event listeners
    if (dropzone) {
        ["dragenter", "dragover"].forEach((eventName) => {
            dropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropzone.classList.add("dragover");
            }, false);
        });

        ["dragleave", "drop"].forEach((eventName) => {
            dropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropzone.classList.remove("dragover");
            }, false);
        });

        dropzone.addEventListener("drop", (e) => {
            const dt = e.dataTransfer;
            if (dt && dt.files && dt.files[0]) {
                if (fileInput) {
                    fileInput.files = dt.files;
                }
                handleFileSelection(dt.files[0]);
            }
        });
    }

    // Form submission with AJAX & Progress Feedback
    if (uploadForm) {
        uploadForm.addEventListener("submit", async (e) => {
            e.preventDefault();
            clearAlert();

            const file = fileInput && fileInput.files ? fileInput.files[0] : null;
            if (!file) {
                showAlert("Please select a PDF document first.", "error");
                return;
            }

            // Client-side validations
            if (!file.name.toLowerCase().endsWith(".pdf")) {
                showAlert("Unsupported file format. Please upload a PDF file.", "error");
                return;
            }

            if (file.size > MAX_FILE_SIZE_BYTES) {
                showAlert("File exceeds the maximum upload limit of 16 MB.", "error");
                return;
            }

            // Show Progress UI
            if (btnSubmitUpload) btnSubmitUpload.disabled = true;
            if (progressContainer) progressContainer.classList.remove("hidden");
            if (progressBarFill) progressBarFill.style.width = "25%";
            if (progressStatusText) progressStatusText.textContent = "Uploading document to server...";

            const formData = new FormData();
            formData.append("file", file);

            try {
                // Animate progress bar toward 70% during network transfer & PyMuPDF processing
                const progressInterval = setInterval(() => {
                    if (progressBarFill) {
                        const current = parseInt(progressBarFill.style.width) || 25;
                        if (current < 85) {
                            progressBarFill.style.width = `${current + 15}%`;
                        }
                    }
                }, 300);

                if (progressStatusText) progressStatusText.textContent = "Validating document & extracting page text with PyMuPDF...";

                const response = await fetch("/upload", {
                    method: "POST",
                    body: formData,
                    headers: {
                        "X-Requested-With": "XMLHttpRequest",
                        "Accept": "application/json"
                    }
                });

                clearInterval(progressInterval);

                const data = await response.json();

                if (!response.ok || !data.success) {
                    throw new Error(data.error || "Upload failed. Please check the document.");
                }

                // Successful processing
                if (progressBarFill) progressBarFill.style.width = "100%";
                if (progressStatusText) progressStatusText.textContent = "Document processed successfully! Redirecting...";

                showAlert(data.message, data.is_scanned ? "warning" : "success");

                // Redirect to document details page
                setTimeout(() => {
                    if (data.redirect_url) {
                        window.location.href = data.redirect_url;
                    }
                }, 750);

            } catch (err) {
                console.error("Upload error:", err);
                showAlert(err.message || "An unexpected error occurred during upload.", "error");

                if (progressContainer) progressContainer.classList.add("hidden");
                if (btnSubmitUpload) btnSubmitUpload.disabled = false;
            }
        });
    }

    // -------------------------------------------------------------------------
    // 4. Documents List Table Search Filtering
    // -------------------------------------------------------------------------
    const docSearchInput = document.getElementById("doc-search-input");
    const documentsTable = document.getElementById("documents-table");
    const tableCountLabel = document.getElementById("table-count-label");

    if (docSearchInput && documentsTable) {
        docSearchInput.addEventListener("input", () => {
            const query = docSearchInput.value.toLowerCase().trim();
            const rows = documentsTable.querySelectorAll("tbody tr");
            let visibleCount = 0;

            rows.forEach((row) => {
                const titleEl = row.querySelector(".doc-link-title");
                const titleText = titleEl ? titleEl.textContent.toLowerCase() : "";
                if (titleText.includes(query)) {
                    row.style.display = "";
                    visibleCount++;
                } else {
                    row.style.display = "none";
                }
            });

            if (tableCountLabel) {
                tableCountLabel.textContent = `Showing ${visibleCount} of ${rows.length} documents`;
            }
        });
    }

    // -------------------------------------------------------------------------
    // 5. Document Details Page Content Search Filtering
    // -------------------------------------------------------------------------
    const pageSearchInput = document.getElementById("page-search-input");
    const pagesList = document.getElementById("pages-list");

    if (pageSearchInput && pagesList) {
        pageSearchInput.addEventListener("input", () => {
            const query = pageSearchInput.value.toLowerCase().trim();
            const pageCards = pagesList.querySelectorAll(".page-card");

            pageCards.forEach((card) => {
                const textEl = card.querySelector(".page-text-block");
                const text = textEl ? textEl.textContent.toLowerCase() : "";
                const pageNumberText = card.querySelector(".page-number-pill") ? card.querySelector(".page-number-pill").textContent.toLowerCase() : "";

                if (query === "" || text.includes(query) || pageNumberText.includes(query)) {
                    card.style.display = "";
                } else {
                    card.style.display = "none";
                }
            });
        });
    }

    // -------------------------------------------------------------------------
    // 6. Copy Page Text Buttons
    // -------------------------------------------------------------------------
    document.querySelectorAll(".btn-copy-page").forEach((btn) => {
        btn.addEventListener("click", async () => {
            const card = btn.closest(".page-card");
            if (!card) return;
            const textBlock = card.querySelector(".page-text-block");
            const textToCopy = textBlock ? textBlock.textContent : "";

            try {
                await navigator.clipboard.writeText(textToCopy);
                const originalHtml = btn.innerHTML;
                btn.innerHTML = `
                    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <polyline points="20 6 9 17 4 12"/>
                    </svg>
                    <span>Copied!</span>
                `;
                btn.style.color = "#34d399";

                setTimeout(() => {
                    btn.innerHTML = originalHtml;
                    btn.style.color = "";
                }, 1800);
            } catch (err) {
                console.error("Failed to copy page text:", err);
            }
        });
    });

    // -------------------------------------------------------------------------
    // 7. Phase 3: AI Summarization & Study Insights Interactivity
    // -------------------------------------------------------------------------
    const summarizationSection = document.getElementById("summarization-section");
    if (summarizationSection) {
        const docId = summarizationSection.getAttribute("data-doc-id");
        const btnGenerate = document.getElementById("btn-generate-summary");
        const btnGenerateText = document.getElementById("btn-generate-text");
        const summaryAlert = document.getElementById("summary-alert");
        const summaryLoading = document.getElementById("summary-loading");
        const summaryProgressBar = document.getElementById("summary-progress-bar");
        const loadingStepTitle = document.getElementById("loading-step-title");
        const loadingStepSub = document.getElementById("loading-step-sub");
        const summaryContentPanel = document.getElementById("summary-content-panel");
        const summaryEmptyPlaceholder = document.getElementById("summary-empty-placeholder");
        const modePills = document.querySelectorAll(".mode-pill");

        // Helper to show inline summary alerts
        function showSummaryAlert(message, type = "error") {
            if (!summaryAlert) return;
            summaryAlert.className = `summary-alert alert alert-${type}`;
            const icon = type === "success" ? "✓" : (type === "warning" ? "⚠️" : "✕");
            summaryAlert.innerHTML = `
                <span class="alert-icon">${icon}</span>
                <span class="alert-message">${message}</span>
                <button type="button" class="alert-close" onclick="this.parentElement.classList.add('hidden')">×</button>
            `;
            summaryAlert.classList.remove("hidden");
        }

        function clearSummaryAlert() {
            if (summaryAlert) summaryAlert.classList.add("hidden");
        }

        // Mode Pill Selection with dynamic summary fetching
        modePills.forEach((pill) => {
            pill.addEventListener("click", async () => {
                modePills.forEach((p) => p.classList.remove("active"));
                pill.classList.add("active");
                const radio = pill.querySelector("input[type='radio']");
                if (radio) radio.checked = true;

                const mode = radio ? radio.value : "standard";
                const modeLabel = mode.charAt(0).toUpperCase() + mode.slice(1);

                const currentDocId = summarizationSection.getAttribute("data-doc-id");
                if (!currentDocId) return;

                // Fetch if summary of this mode already exists
                try {
                    const res = await fetch(`/document/${currentDocId}/summary?type=${mode}`, {
                        headers: { "Accept": "application/json" }
                    });
                    if (res.ok) {
                        const data = await res.json();
                        if (data.success && data.summary_text) {
                            renderSummaryData(data);
                            if (btnGenerateText) btnGenerateText.textContent = "Regenerate Summary";
                            return;
                        }
                    }
                } catch (e) {
                    // Ignore network check errors
                }

                // If not found, show empty placeholder for this mode
                if (summaryContentPanel) summaryContentPanel.classList.add("hidden");
                if (summaryEmptyPlaceholder) {
                    summaryEmptyPlaceholder.classList.remove("hidden");
                    const title = summaryEmptyPlaceholder.querySelector("h3");
                    if (title) title.textContent = `No ${modeLabel} Summary Generated Yet`;
                }
                if (btnGenerateText) btnGenerateText.textContent = "Generate Summary";
            });
        });

        // Event Delegation for Summary Tabs and Copy Button (Zero Duplicate Listeners)
        summarizationSection.addEventListener("click", (e) => {
            // Tab button click
            const tabBtn = e.target.closest(".summary-tab");
            if (tabBtn) {
                const targetId = tabBtn.getAttribute("data-tab");
                const tabs = summarizationSection.querySelectorAll(".summary-tab");
                const panes = summarizationSection.querySelectorAll(".tab-pane");

                tabs.forEach((t) => {
                    t.classList.remove("active");
                    t.setAttribute("aria-selected", "false");
                    t.style.borderBottom = "none";
                    t.style.color = "var(--text-secondary)";
                });
                panes.forEach((p) => {
                    p.classList.add("hidden");
                    p.classList.remove("active");
                });

                tabBtn.classList.add("active");
                tabBtn.setAttribute("aria-selected", "true");
                tabBtn.style.borderBottom = "3px solid var(--terracotta)";
                tabBtn.style.color = "var(--text-primary)";

                const targetPane = document.getElementById(targetId);
                if (targetPane) {
                    targetPane.classList.remove("hidden");
                    targetPane.classList.add("active");
                }
                return;
            }

            // Copy button click
            const btnCopy = e.target.closest("#btn-copy-summary, .btn-copy-summary");
            if (btnCopy) {
                const proseContainer = document.getElementById("summary-prose-container");
                if (!proseContainer) return;
                const text = proseContainer.innerText;

                navigator.clipboard.writeText(text).then(() => {
                    const originalHtml = btnCopy.innerHTML;
                    btnCopy.innerHTML = "<span>✓ Copied Summary!</span>";
                    btnCopy.style.color = "var(--forest-green)";
                    setTimeout(() => {
                        btnCopy.innerHTML = originalHtml;
                        btnCopy.style.color = "";
                    }, 2000);
                }).catch((err) => {
                    console.error("Failed to copy summary:", err);
                });
            }
        });

        // Render Summary Data into DOM dynamically
        function renderSummaryData(data) {
            clearSummaryAlert();
            if (summaryEmptyPlaceholder) summaryEmptyPlaceholder.classList.add("hidden");
            if (summaryContentPanel) summaryContentPanel.classList.remove("hidden");

            // Format metrics
            const totalWords = data.stats?.original_words || parseInt(document.querySelector(".metrics-grid .metric-item:nth-child(2) .metric-value")?.textContent.replace(/,/g, "") || "0", 10);
            const summaryWords = data.stats?.summary_words || (data.summary_text ? data.summary_text.split(/\s+/).length : 0);
            const redPct = data.stats?.reduction_percentage !== undefined ? data.stats.reduction_percentage : (totalWords > 0 ? Math.max(0, Math.round((1 - summaryWords / totalWords) * 100 * 10) / 10) : 0);

            const modeLabel = data.summary_type ? data.summary_type.charAt(0).toUpperCase() + data.summary_type.slice(1) : "Standard";
            const modelLabel = data.model_name || "distilbart-cnn-12-6";

            // Paragraphs HTML
            const paragraphs = (data.summary_text || "").split(/\n\n+/).filter((p) => p.trim());
            const proseHtml = paragraphs.map((p) => {
                if (p.startsWith("Key Points:")) {
                    const lines = p.split("\n");
                    const title = lines[0];
                    const pts = lines.slice(1).map(l => `<li style="margin-bottom:6px;">${l.replace(/^•\s*/, "")}</li>`).join("");
                    return `<div style="margin-top:16px; background:var(--bg-secondary); padding:16px 20px; border-radius:var(--radius-sm); border-left:3px solid var(--accent-terracotta);"><strong style="color:var(--text-primary); display:block; margin-bottom:8px;">${title}</strong><ul style="margin:0; padding-left:20px; color:var(--text-primary);">${pts}</ul></div>`;
                }
                return `<p style="margin-bottom:16px; line-height:1.75;">${p.trim()}</p>`;
            }).join("");

            // Concepts HTML
            const concepts = data.key_concepts || [];
            const conceptsHtml = concepts.length > 0
                ? concepts.map((c) => `
                    <div class="concept-card" style="background:#fff; border:1px solid var(--border-warm); border-left:4px solid var(--terracotta); border-radius:var(--radius-sm); padding:18px 20px;">
                        <div class="concept-term-header" style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">
                            <span class="concept-badge" style="font-size:0.7rem; font-weight:700; background:rgba(201,111,72,0.15); color:var(--terracotta); padding:2px 8px; border-radius:10px; text-transform:uppercase;">Concept</span>
                            <h4 class="concept-term" style="font-family:var(--font-heading); font-size:1.15rem; color:var(--text-primary); margin:0;">${c.term}</h4>
                        </div>
                        <p class="concept-definition" style="font-size:0.9rem; color:var(--text-secondary); margin:0; line-height:1.6;">${c.definition}</p>
                    </div>
                `).join("")
                : '<p class="text-muted" style="color:var(--text-secondary);">No explicit definition patterns identified in this text.</p>';

            // Exam Points HTML
            const examPoints = data.exam_points || [];
            const examHtml = examPoints.length > 0
                ? examPoints.map((pt, idx) => `
                    <div class="exam-point-item" style="display:flex; gap:14px; background:#fff; border:1px solid var(--border-warm); border-radius:var(--radius-sm); padding:14px 18px; align-items:flex-start;">
                        <div class="exam-point-num" style="background:var(--terracotta); color:#fff; font-weight:800; font-size:0.85rem; width:26px; height:26px; border-radius:50%; display:flex; align-items:center; justify-content:center; flex-shrink:0;">${idx + 1}</div>
                        <div class="exam-point-text" style="font-size:0.95rem; color:var(--text-primary); line-height:1.6;">${pt}</div>
                    </div>
                `).join("")
                : '<p class="text-muted" style="color:var(--text-secondary);">No exam-specific indicator sentences detected.</p>';

            // Keywords HTML
            const keywords = data.keywords || [];
            const keywordsHtml = keywords.length > 0
                ? keywords.map((kw) => `
                    <span class="keyword-chip" style="background:#fff; border:1px solid var(--border-warm); padding:6px 14px; border-radius:16px; font-size:0.85rem; color:var(--forest-green); font-weight:600; cursor:pointer;" title="Click to copy" onclick="navigator.clipboard.writeText('${kw}')">${kw}</span>
                `).join("")
                : '<p class="text-muted" style="color:var(--text-secondary);">No keywords extracted.</p>';

            // Chapters HTML
            const chapters = data.chapter_summaries || [];
            const chaptersHtml = chapters.length > 0
                ? chapters.map((item) => `
                    <div class="chapter-card" style="background:#fff; border:1px solid var(--border-warm); border-radius:var(--radius-sm); padding:16px 20px;">
                        <div class="chapter-card-header" style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                            <span class="chapter-pill" style="background:rgba(130,150,107,0.18); font-weight:700; font-size:0.8rem; padding:4px 10px; border-radius:12px; color:var(--forest-green);">${item.section}</span>
                            <span class="font-mono text-muted" style="font-family:monospace; font-size:0.8rem; color:var(--text-secondary);">${item.word_count || (item.summary ? item.summary.split(/\s+/).length : 0)} words</span>
                        </div>
                        <p class="chapter-summary-text" style="font-size:0.92rem; color:var(--text-primary); margin:0; line-height:1.6;">${item.summary}</p>
                    </div>
                `).join("")
                : '<p class="text-muted" style="color:var(--text-secondary);">No chapter breakdowns available.</p>';

            // Topics HTML
            const topics = data.topics || [];
            const topicsHtml = topics.length > 0
                ? topics.map((t) => `
                    <div class="topic-item" style="background:#fff; border:1px solid var(--border-warm); border-radius:var(--radius-sm); padding:18px 20px;">
                        <h4 class="topic-heading" style="font-family:var(--font-heading); font-size:1.15rem; color:var(--text-primary); margin:0 0 10px 0;">${t.topic}</h4>
                        ${t.subtopics && t.subtopics.length > 0 ? `
                            <div class="subtopics-list" style="margin:0; padding-left:20px; color:var(--text-secondary); font-size:0.9rem; line-height:1.8;">
                                ${t.subtopics.map((s) => `
                                    <div class="subtopic-item">
                                        <span class="subtopic-bullet">•</span>
                                        <span class="subtopic-text">${s}</span>
                                    </div>
                                `).join("")}
                            </div>
                        ` : ""}
                    </div>
                `).join("")
                : '<p class="text-muted" style="color:var(--text-secondary);">No topics outline detected.</p>';

            // Rebuild summary panel contents
            summaryContentPanel.innerHTML = `
                <div class="summary-meta-strip" style="display:flex; gap:16px; flex-wrap:wrap; align-items:center; background:var(--bg-secondary); border:1px solid var(--border-warm); border-radius:var(--radius-sm); padding:10px 16px; margin-bottom:20px;">
                    <div class="summary-stat-item" style="font-size:0.85rem;">
                        <span class="stat-label" style="color:var(--text-secondary);">Mode:</span>
                        <strong style="color:var(--forest-green); text-transform:capitalize;" id="stat-summary-mode">${modeLabel}</strong>
                    </div>
                    <div class="summary-stat-item" style="font-size:0.85rem;">
                        <span class="stat-label" style="color:var(--text-secondary);">Model:</span>
                        <span class="font-mono text-muted" style="font-family:monospace; color:var(--text-primary);" id="stat-summary-model">${modelLabel}</span>
                    </div>
                    <div class="summary-stat-item" style="font-size:0.85rem;">
                        <span class="stat-label" style="color:var(--text-secondary);">Summary Length:</span>
                        <strong id="stat-summary-words">${summaryWords} words</strong>
                    </div>
                    ${redPct > 0 ? `
                        <div class="summary-stat-item" style="font-size:0.85rem; background:rgba(130,150,107,0.2); color:var(--forest-green); padding:2px 10px; border-radius:12px; font-weight:700;">
                            ⏱️ ~${redPct}% reading time saved
                        </div>
                    ` : ""}
                    <div class="summary-stat-item" style="margin-left:auto;">
                        <button type="button" class="btn-copy-summary" id="btn-copy-summary" style="background:#fff; border:1px solid var(--border-warm); padding:6px 14px; border-radius:16px; font-size:0.8rem; cursor:pointer; color:var(--text-primary); font-weight:600;" title="Copy full summary">
                            📋 <span>Copy Summary</span>
                        </button>
                    </div>
                </div>

                <div class="summary-tabs" role="tablist" style="display:flex; gap:8px; border-bottom:1.5px solid var(--border-warm); margin-bottom:20px; overflow-x:auto;">
                    <button type="button" class="summary-tab active" role="tab" aria-selected="true" id="tab-btn-summary-text" data-tab="tab-summary-text" style="background:transparent; border:none; border-bottom:3px solid var(--terracotta); padding:10px 16px; font-weight:700; color:var(--text-primary); cursor:pointer;">
                        <span>📝 Summary</span>
                    </button>
                    <button type="button" class="summary-tab" role="tab" aria-selected="false" id="tab-btn-key-concepts" data-tab="tab-key-concepts" style="background:transparent; border:none; padding:10px 16px; font-weight:600; color:var(--text-secondary); cursor:pointer;">
                        <span>💡 Key Concepts (${concepts.length})</span>
                    </button>
                    <button type="button" class="summary-tab" role="tab" aria-selected="false" id="tab-btn-exam-points" data-tab="tab-exam-points" style="background:transparent; border:none; padding:10px 16px; font-weight:600; color:var(--text-secondary); cursor:pointer;">
                        <span>🎯 Exam Points (${examPoints.length})</span>
                    </button>
                    <button type="button" class="summary-tab" role="tab" aria-selected="false" id="tab-btn-keywords" data-tab="tab-keywords" style="background:transparent; border:none; padding:10px 16px; font-weight:600; color:var(--text-secondary); cursor:pointer;">
                        <span>🏷️ Keywords (${keywords.length})</span>
                    </button>
                    ${chapters.length > 0 ? `
                        <button type="button" class="summary-tab" role="tab" aria-selected="false" id="tab-btn-chapters" data-tab="tab-chapters" style="background:transparent; border:none; padding:10px 16px; font-weight:600; color:var(--text-secondary); cursor:pointer;">
                            <span>📖 Chapter Breakdown (${chapters.length})</span>
                        </button>
                    ` : ""}
                    ${topics.length > 0 ? `
                        <button type="button" class="summary-tab" role="tab" aria-selected="false" id="tab-btn-topics" data-tab="tab-topics" style="background:transparent; border:none; padding:10px 16px; font-weight:600; color:var(--text-secondary); cursor:pointer;">
                            <span>📂 Topics Outline</span>
                        </button>
                    ` : ""}
                </div>

                <div class="tab-pane active" id="tab-summary-text" role="tabpanel">
                    <div style="background:#fff; border:1px solid var(--border-warm); border-radius:var(--radius-md); padding:28px 32px; box-shadow:var(--shadow-sm); line-height:1.8; font-size:1.05rem; color:var(--text-primary);">
                        <div class="summary-prose" id="summary-prose-container">
                            ${proseHtml}
                        </div>
                    </div>
                </div>

                <div class="tab-pane hidden" id="tab-key-concepts" role="tabpanel">
                    <div class="concepts-grid" id="concepts-grid-container" style="display:grid; grid-template-columns:repeat(auto-fit, minmax(280px, 1fr)); gap:16px;">
                        ${conceptsHtml}
                    </div>
                </div>

                <div class="tab-pane hidden" id="tab-exam-points" role="tabpanel">
                    <div class="exam-points-list" id="exam-points-container" style="display:flex; flex-direction:column; gap:12px;">
                        ${examHtml}
                    </div>
                </div>

                <div class="tab-pane hidden" id="tab-keywords" role="tabpanel">
                    <div class="keywords-chips-container" id="keywords-chips-container" style="display:flex; flex-wrap:wrap; gap:10px;">
                        ${keywordsHtml}
                    </div>
                </div>

                <div class="tab-pane hidden" id="tab-chapters" role="tabpanel">
                    <div class="chapters-list" id="chapters-list-container" style="display:flex; flex-direction:column; gap:14px;">
                        ${chaptersHtml}
                    </div>
                </div>

                <div class="tab-pane hidden" id="tab-topics" role="tabpanel">
                    <div class="topics-outline-list" id="topics-outline-container" style="display:flex; flex-direction:column; gap:14px;">
                        ${topicsHtml}
                    </div>
                </div>
            `;
        }

        // Generate Summary Action
        if (btnGenerate) {
            btnGenerate.addEventListener("click", async () => {
                clearSummaryAlert();

                // Get selected radio mode
                const selectedRadio = summarizationSection.querySelector("input[name='summary_type_select']:checked");
                const summaryType = selectedRadio ? selectedRadio.value : "standard";

                // UI loading state
                btnGenerate.disabled = true;
                if (btnGenerateText) btnGenerateText.textContent = "Generating Summary...";
                if (summaryLoading) summaryLoading.classList.remove("hidden");
                if (summaryProgressBar) summaryProgressBar.style.width = "10%";

                const currentDocId = summarizationSection.getAttribute("data-doc-id");
                if (!currentDocId) {
                    showSummaryAlert("Please select an uploaded document before generating a summary.", "warning");
                    btnGenerate.disabled = false;
                    return;
                }

                // Animate progress steps
                let progressPercent = 10;
                const progressInterval = setInterval(() => {
                    progressPercent = Math.min(progressPercent + 12, 92);
                    if (summaryProgressBar) summaryProgressBar.style.width = `${progressPercent}%`;

                    if (progressPercent > 70 && loadingStepTitle) {
                        loadingStepTitle.textContent = "Synthesizing Concepts & Keywords...";
                        if (loadingStepSub) loadingStepSub.textContent = "Running TF-IDF keyword extraction and definition pattern analysis.";
                    } else if (progressPercent > 35 && loadingStepTitle) {
                        loadingStepTitle.textContent = "AI Model Running Inference...";
                        if (loadingStepSub) loadingStepSub.textContent = "HuggingFace DistilBART Transformer generating abstractive summaries.";
                    }
                }, 600);

                try {
                    const response = await fetch(`/document/${currentDocId}/summarize`, {
                        method: "POST",
                        headers: {
                            "Content-Type": "application/json",
                            "X-Requested-With": "XMLHttpRequest",
                            "Accept": "application/json",
                        },
                        body: JSON.stringify({ summary_type: summaryType }),
                    });

                    clearInterval(progressInterval);

                    const data = await response.json();

                    if (!response.ok || !data.success) {
                        throw new Error(data.error || "Summarization failed. Please check the document.");
                    }

                    if (summaryProgressBar) summaryProgressBar.style.width = "100%";
                    if (loadingStepTitle) loadingStepTitle.textContent = "Summary Generated Successfully!";

                    setTimeout(() => {
                        if (summaryLoading) summaryLoading.classList.add("hidden");
                        renderSummaryData(data);
                        showSummaryAlert(data.message, "success");
                        btnGenerate.disabled = false;
                        if (btnGenerateText) btnGenerateText.textContent = "Regenerate Summary";
                    }, 400);

                } catch (err) {
                    clearInterval(progressInterval);
                    console.error("Summarization error:", err);
                    if (summaryLoading) summaryLoading.classList.add("hidden");
                    showSummaryAlert(err.message || "An unexpected error occurred during summarization.", "error");
                    btnGenerate.disabled = false;
                    if (btnGenerateText) btnGenerateText.textContent = "Generate Summary";
                }
            });
        }
    }

    // -------------------------------------------------------------------------
    // 6. Phase 4 - AI Study Chatbot & RAG Controller
    // -------------------------------------------------------------------------
    const chatWindowCard = document.getElementById("chat-window-card");
    if (chatWindowCard) {
        const docId = chatWindowCard.getAttribute("data-doc-id");
        const convId = chatWindowCard.getAttribute("data-conv-id");
        const messagesContainer = document.getElementById("chat-messages-container");
        const questionInput = document.getElementById("chat-question-input");
        const btnSend = document.getElementById("btn-chat-send");
        const typingRow = document.getElementById("chat-typing-row");
        const typingStatusText = document.getElementById("typing-status-text");
        const suggestedChipsContainer = document.getElementById("suggested-chips-container");
        const btnClearChat = document.getElementById("btn-clear-chat");
        const btnReindex = document.getElementById("btn-reindex");
        const chatAlertContainer = document.getElementById("chat-alert-container");
        const docChunksCount = document.getElementById("doc-chunks-count");
        const indexStatusBadge = document.getElementById("index-status-badge");

        let isWaitingForResponse = false;

        function scrollToBottom() {
            if (messagesContainer) {
                messagesContainer.scrollTo({
                    top: messagesContainer.scrollHeight,
                    behavior: "smooth",
                });
            }
        }
        // Initial scroll to bottom on load
        setTimeout(scrollToBottom, 150);

        function showChatAlert(message, type = "error") {
            if (!chatAlertContainer) return;
            chatAlertContainer.className = `alert alert-${type}`;
            const icon = type === "success" ? "✓" : (type === "warning" ? "⚠️" : "✕");
            chatAlertContainer.innerHTML = `
                <span class="alert-icon">${icon}</span>
                <span class="alert-message">${message}</span>
                <button type="button" class="alert-close" onclick="this.parentElement.classList.add('hidden')">×</button>
            `;
            chatAlertContainer.classList.remove("hidden");
        }

        function clearChatAlert() {
            if (chatAlertContainer) {
                chatAlertContainer.classList.add("hidden");
                chatAlertContainer.innerHTML = "";
            }
        }

        function formatTimeNow() {
            const now = new Date();
            return now.toTimeString().substring(0, 5);
        }

        function escapeHtml(text) {
            if (!text) return "";
            const div = document.createElement("div");
            div.textContent = text;
            return div.innerHTML;
        }

        function appendUserMessage(text) {
            const row = document.createElement("div");
            row.className = "chat-message-row user-row";
            row.innerHTML = `
                <div class="chat-bubble user-bubble">
                    <div class="bubble-header">
                        <span class="sender-name">You</span>
                        <span class="bubble-timestamp font-mono">${formatTimeNow()}</span>
                    </div>
                    <div class="bubble-body">
                        <p>${escapeHtml(text)}</p>
                    </div>
                </div>
                <div class="chat-avatar user-avatar">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>
                        <circle cx="12" cy="7" r="4"/>
                    </svg>
                </div>
            `;
            if (typingRow) {
                messagesContainer.insertBefore(row, typingRow);
            } else {
                messagesContainer.appendChild(row);
            }
            scrollToBottom();
        }

        function appendAssistantMessage(data) {
            const row = document.createElement("div");
            row.className = "chat-message-row assistant-row";

            const sources = data.sources || [];
            const isGrounded = data.is_grounded !== false && sources.length > 0;
            const badgeClass = isGrounded ? "badge-success" : "badge-warning";
            const badgeText = isGrounded
                ? `Grounded (${sources.length} Citation${sources.length > 1 ? "s" : ""})`
                : "No Direct Excerpt";

            let sourcesHtml = "";
            if (sources.length > 0) {
                const sourceCards = sources.map(src => {
                    const scoreText = src.similarity_score
                        ? `<span class="source-score font-mono">${Math.round(src.similarity_score * 1000) / 10}% match</span>`
                        : "";
                    return `
                        <div class="source-card">
                            <div class="source-card-header">
                                <span class="source-page-badge font-mono">Page ${src.page_number}</span>
                                <span class="source-filename">${escapeHtml(src.original_filename || "")}</span>
                                ${scoreText}
                            </div>
                            <div class="source-card-excerpt">
                                &ldquo;${escapeHtml(src.excerpt || "")}&rdquo;
                            </div>
                        </div>
                    `;
                }).join("");

                sourcesHtml = `
                    <div class="sources-accordion">
                        <div class="sources-header">
                            <span class="sources-title">
                                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                                    <polyline points="14 2 14 8 20 8"/>
                                </svg>
                                Verified Citations (${sources.length})
                            </span>
                        </div>
                        <div class="sources-cards-list">
                            ${sourceCards}
                        </div>
                    </div>
                `;
            }

            row.innerHTML = `
                <div class="chat-avatar assistant-avatar">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
                    </svg>
                </div>
                <div class="chat-bubble assistant-bubble">
                    <div class="bubble-header">
                        <span class="sender-name">Study Assistant AI</span>
                        <span class="badge-status ${badgeClass}">${badgeText}</span>
                        <span class="bubble-timestamp font-mono">${formatTimeNow()}</span>
                    </div>
                    <div class="bubble-body">
                        <p>${escapeHtml(data.answer || "")}</p>
                        ${sourcesHtml}
                    </div>
                </div>
            `;

            if (typingRow) {
                messagesContainer.insertBefore(row, typingRow);
            } else {
                messagesContainer.appendChild(row);
            }
            scrollToBottom();

            // Update suggested chips if returned
            if (data.suggested_questions && data.suggested_questions.length > 0) {
                renderSuggestedChips(data.suggested_questions);
            }
        }

        function renderSuggestedChips(questions) {
            if (!suggestedChipsContainer) return;
            suggestedChipsContainer.innerHTML = "";
            questions.forEach(q => {
                const chip = document.createElement("button");
                chip.type = "button";
                chip.className = "suggested-chip";
                chip.setAttribute("data-question", q);
                chip.innerHTML = `<span>${escapeHtml(q)}</span>`;
                chip.addEventListener("click", () => {
                    if (isWaitingForResponse) return;
                    sendQuestion(q);
                });
                suggestedChipsContainer.appendChild(chip);
            });
        }

        // Bind existing starter chips
        if (suggestedChipsContainer) {
            const existingChips = suggestedChipsContainer.querySelectorAll(".suggested-chip");
            existingChips.forEach(chip => {
                chip.addEventListener("click", () => {
                    const q = chip.getAttribute("data-question");
                    if (q && !isWaitingForResponse) {
                        sendQuestion(q);
                    }
                });
            });
        }

        // Auto-expand input textarea and keyboard handlers
        if (questionInput) {
            questionInput.addEventListener("input", () => {
                questionInput.style.height = "auto";
                questionInput.style.height = `${Math.min(questionInput.scrollHeight, 140)}px`;
            });

            questionInput.addEventListener("keydown", (e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    const text = questionInput.value.trim();
                    if (text && !isWaitingForResponse) {
                        sendQuestion(text);
                    }
                }
            });
        }

        // Send Button
        if (btnSend) {
            btnSend.addEventListener("click", () => {
                if (isWaitingForResponse) return;
                const text = questionInput ? questionInput.value.trim() : "";
                if (text) {
                    sendQuestion(text);
                }
            });
        }

        async function sendQuestion(text) {
            if (!text || isWaitingForResponse) return;
            clearChatAlert();

            // Set loading state
            isWaitingForResponse = true;
            if (btnSend) btnSend.disabled = true;
            if (questionInput) {
                questionInput.value = "";
                questionInput.style.height = "auto";
                questionInput.disabled = true;
            }

            // Append user message to thread
            appendUserMessage(text);

            // Show typing indicator
            if (typingRow) {
                typingRow.classList.remove("hidden");
                if (typingStatusText) {
                    typingStatusText.textContent = "Searching document chunks & generating grounded answer...";
                }
            }
            scrollToBottom();

            try {
                const response = await fetch(`/document/${docId}/chat/ask`, {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                        "X-Requested-With": "XMLHttpRequest",
                        "Accept": "application/json",
                    },
                    body: JSON.stringify({ question: text }),
                });

                const data = await response.json();

                if (!response.ok || !data.success) {
                    throw new Error(data.error || "Failed to get answer from study assistant.");
                }

                // Append assistant reply
                if (typingRow) typingRow.classList.add("hidden");
                appendAssistantMessage(data);

            } catch (err) {
                console.error("Chat error:", err);
                if (typingRow) typingRow.classList.add("hidden");
                showChatAlert(err.message || "An error occurred while answering your question.", "error");

                appendAssistantMessage({
                    answer: "Sorry, I could not answer this question: " + (err.message || "Please check your network connection and try again."),
                    sources: [],
                    is_grounded: false,
                    suggested_questions: [
                        "What is the main topic of this document?",
                        "Can you summarize this document for me?",
                    ],
                });
            } finally {
                isWaitingForResponse = false;
                if (btnSend) btnSend.disabled = false;
                if (questionInput) {
                    questionInput.disabled = false;
                    questionInput.focus();
                }
            }
        }

        // Clear Chat Button
        if (btnClearChat) {
            btnClearChat.addEventListener("click", async () => {
                if (!confirm("Are you sure you want to clear the conversation history for this document?")) {
                    return;
                }
                clearChatAlert();
                try {
                    const response = await fetch(`/document/${docId}/chat/clear`, {
                        method: "POST",
                        headers: {
                            "Content-Type": "application/json",
                            "X-Requested-With": "XMLHttpRequest",
                            "Accept": "application/json",
                        },
                    });
                    const data = await response.json();
                    if (!response.ok || !data.success) {
                        throw new Error(data.error || "Failed to clear conversation.");
                    }

                    // Reset messages list to just the welcome row
                    if (messagesContainer) {
                        const welcomeRow = messagesContainer.querySelector(".welcome-row");
                        messagesContainer.innerHTML = "";
                        if (welcomeRow) {
                            messagesContainer.appendChild(welcomeRow);
                        }
                    }
                    showChatAlert("Conversation history cleared successfully.", "success");
                    setTimeout(clearChatAlert, 3000);

                } catch (err) {
                    showChatAlert(err.message || "Could not clear chat history.", "error");
                }
            });
        }

        // Re-Index Button
        if (btnReindex) {
            btnReindex.addEventListener("click", async () => {
                clearChatAlert();
                btnReindex.disabled = true;
                const origHtml = btnReindex.innerHTML;
                btnReindex.innerHTML = `
                    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="spinning">
                        <path d="M21 12a9 9 0 1 1-6.219-8.56"/>
                    </svg>
                    <span>Reindexing...</span>
                `;

                try {
                    const response = await fetch(`/document/${docId}/index`, {
                        method: "POST",
                        headers: {
                            "Content-Type": "application/json",
                            "X-Requested-With": "XMLHttpRequest",
                            "Accept": "application/json",
                        },
                        body: JSON.stringify({ force_reindex: true }),
                    });
                    const data = await response.json();
                    if (!response.ok || !data.success) {
                        throw new Error(data.error || "Failed to reindex document.");
                    }

                    const chunks = data.result?.total_chunks || 0;
                    if (docChunksCount) docChunksCount.textContent = chunks;
                    if (indexStatusBadge) {
                        indexStatusBadge.textContent = "Indexed in FAISS";
                        indexStatusBadge.className = "badge-status badge-success";
                    }

                    showChatAlert(`Document successfully re-indexed! (${chunks} vector chunks indexed in FAISS)`, "success");
                    setTimeout(clearChatAlert, 4000);
                } catch (err) {
                    showChatAlert(err.message || "Reindexing failed.", "error");
                } finally {
                    btnReindex.disabled = false;
                    btnReindex.innerHTML = origHtml;
                }
            });
        }
    }

    // =========================================================================
    // 7. Phase 5: Interactive AI Quiz Controller
    // =========================================================================
    const quizTakingBody = document.querySelector(".quiz-taking-body");
    if (quizTakingBody) {
        const quizId = quizTakingBody.dataset.quizId;
        const totalQuestions = parseInt(quizTakingBody.dataset.totalQuestions || "0", 10);
        const timeLimitMinutes = parseInt(quizTakingBody.dataset.timeLimit || "0", 10);

        // Load question data
        let questions = [];
        const rawJsonElem = document.getElementById("quiz-questions-json");
        if (rawJsonElem) {
            try {
                questions = JSON.parse(rawJsonElem.textContent);
            } catch (e) {
                console.error("Error parsing embedded quiz questions JSON:", e);
            }
        }

        // Quiz State
        let currentIndex = 0;
        let answers = {};
        let isSubmitting = false;
        const quizStartTime = Date.now();

        // Restore saved answers from localStorage if available
        const storageKey = `quiz_answers_${quizId}`;
        try {
            const saved = localStorage.getItem(storageKey);
            if (saved) {
                answers = JSON.parse(saved);
            }
        } catch (e) {
            answers = {};
        }

        // DOM elements
        const currentQNum = document.getElementById("current-q-num");
        const answeredCountLabel = document.getElementById("answered-count-label");
        const answeredPctLabel = document.getElementById("answered-pct-label");
        const progressFill = document.getElementById("quiz-progress-fill");
        const qNumberBadge = document.getElementById("q-number-badge");
        const qSourceBadge = document.getElementById("q-source-badge");
        const questionText = document.getElementById("question-text");
        const optionsContainer = document.getElementById("options-container");
        const btnPrev = document.getElementById("btn-prev-question");
        const btnNext = document.getElementById("btn-next-question");
        const btnOpenSubmit = document.getElementById("btn-open-submit-modal");
        const btnSubmitSidebar = document.getElementById("btn-submit-sidebar");
        const submitModal = document.getElementById("submit-confirm-modal");
        const submittingModal = document.getElementById("submitting-modal");
        const btnCancelModal = document.getElementById("btn-cancel-modal");
        const btnConfirmSubmit = document.getElementById("btn-confirm-submit");
        const modalSummaryText = document.getElementById("modal-summary-text");
        const unansweredWarningBox = document.getElementById("unanswered-warning-box");

        // Helper to update progress counters
        function updateProgress() {
            const answeredCount = Object.keys(answers).length;
            const pct = totalQuestions > 0 ? Math.round((answeredCount / totalQuestions) * 100) : 0;

            if (answeredCountLabel) answeredCountLabel.textContent = answeredCount;
            if (answeredPctLabel) answeredPctLabel.textContent = `${pct}%`;
            if (progressFill) progressFill.style.width = `${pct}%`;

            // Update navigator grid dots
            for (let i = 0; i < totalQuestions; i++) {
                const btn = document.getElementById(`q-nav-btn-${i}`);
                if (!btn) continue;
                const q = questions[i];
                const key = q ? (q.id || (i + 1)) : (i + 1);
                if (answers[key]) {
                    btn.classList.add("answered");
                } else {
                    btn.classList.remove("answered");
                }
                if (i === currentIndex) {
                    btn.classList.add("active");
                } else {
                    btn.classList.remove("active");
                }
            }
        }

        // Render Question at index
        function renderQuestion(index) {
            if (!questions || index < 0 || index >= questions.length) return;
            currentIndex = index;
            const q = questions[index];
            const qKey = q.id || (index + 1);

            if (currentQNum) currentQNum.textContent = index + 1;
            if (qNumberBadge) qNumberBadge.textContent = `Q${index + 1}`;
            if (qSourceBadge) {
                let srcText = "Study Material";
                if (q.source_page) {
                    srcText += ` • Page ${q.source_page}`;
                }
                qSourceBadge.textContent = srcText;
            }
            if (questionText) questionText.textContent = q.question_text;

            // Render options text & radio state
            const currentSelected = answers[qKey] || null;
            const letters = ["A", "B", "C", "D"];
            letters.forEach((letter) => {
                const label = document.querySelector(`.option-label[data-option="${letter}"]`);
                const radio = label ? label.querySelector("input[type='radio']") : null;
                const textElem = document.getElementById(`opt-${letter.toLowerCase()}-text`);
                const optValue = q[`option_${letter.toLowerCase()}`];

                if (textElem && optValue !== undefined) {
                    textElem.textContent = optValue;
                }
                if (radio) {
                    radio.checked = (currentSelected === letter);
                }
                if (label) {
                    if (currentSelected === letter) {
                        label.classList.add("selected");
                    } else {
                        label.classList.remove("selected");
                    }
                }
            });

            // Prev / Next button states
            if (btnPrev) {
                btnPrev.disabled = (index === 0);
            }
            if (btnNext) {
                if (index === questions.length - 1) {
                    btnNext.innerHTML = `<span>Review & Submit</span><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>`;
                } else {
                    btnNext.innerHTML = `<span>Next</span><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="9 18 15 12 9 6"/></svg>`;
                }
            }

            updateProgress();
        }

        // Option Selection Event Handling
        if (optionsContainer) {
            optionsContainer.addEventListener("click", (e) => {
                const label = e.target.closest(".option-label");
                if (!label) return;
                const selectedLetter = label.dataset.option;
                if (!selectedLetter) return;

                const q = questions[currentIndex];
                const qKey = q ? (q.id || (currentIndex + 1)) : (currentIndex + 1);
                answers[qKey] = selectedLetter;

                // Persist to local storage
                try {
                    localStorage.setItem(storageKey, JSON.stringify(answers));
                } catch (err) {
                    // Ignore storage quota
                }

                renderQuestion(currentIndex);
            });
        }

        // Next / Prev Button Listeners
        if (btnPrev) {
            btnPrev.addEventListener("click", () => {
                if (currentIndex > 0) {
                    renderQuestion(currentIndex - 1);
                }
            });
        }
        if (btnNext) {
            btnNext.addEventListener("click", () => {
                if (currentIndex < questions.length - 1) {
                    renderQuestion(currentIndex + 1);
                } else {
                    openSubmitModal();
                }
            });
        }

        // Navigator Grid Button Listeners
        const gridContainer = document.getElementById("question-numbers-grid");
        if (gridContainer) {
            gridContainer.addEventListener("click", (e) => {
                const btn = e.target.closest(".q-grid-btn");
                if (!btn) return;
                const idx = parseInt(btn.dataset.index, 10);
                if (!isNaN(idx)) {
                    renderQuestion(idx);
                }
            });
        }

        // Modal Open / Cancel
        function openSubmitModal() {
            const answeredCount = Object.keys(answers).length;
            const unanswered = totalQuestions - answeredCount;

            if (modalSummaryText) {
                modalSummaryText.textContent = `You have answered ${answeredCount} of ${totalQuestions} questions.`;
            }
            if (unansweredWarningBox) {
                if (unanswered > 0) {
                    unansweredWarningBox.classList.remove("hidden");
                } else {
                    unansweredWarningBox.classList.add("hidden");
                }
            }
            if (submitModal) submitModal.classList.remove("hidden");
        }

        if (btnOpenSubmit) btnOpenSubmit.addEventListener("click", openSubmitModal);
        if (btnSubmitSidebar) btnSubmitSidebar.addEventListener("click", openSubmitModal);
        if (btnCancelModal) {
            btnCancelModal.addEventListener("click", () => {
                if (submitModal) submitModal.classList.add("hidden");
            });
        }

        // Submit Quiz Function
        async function submitQuiz(autoTimeout = false) {
            if (isSubmitting) return;
            isSubmitting = true;

            if (submitModal) submitModal.classList.add("hidden");
            if (submittingModal) submittingModal.classList.remove("hidden");

            const timeTaken = Math.max(1, Math.round((Date.now() - quizStartTime) / 1000));
            const payload = {
                answers: answers,
                time_taken_seconds: timeTaken,
                auto_submitted: autoTimeout,
            };

            try {
                const response = await fetch(`/api/quiz/${quizId}/submit`, {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                    },
                    body: JSON.stringify(payload),
                });

                const data = await response.json();
                if (response.ok && data.success && data.redirect_url) {
                    // Clear cached answers
                    try {
                        localStorage.removeItem(storageKey);
                    } catch (e) {}
                    window.location.href = data.redirect_url;
                } else {
                    if (submittingModal) submittingModal.classList.add("hidden");
                    isSubmitting = false;
                    alert(data.error || "Failed to submit quiz. Please try again.");
                }
            } catch (err) {
                if (submittingModal) submittingModal.classList.add("hidden");
                isSubmitting = false;
                alert("Network or server error while submitting quiz.");
            }
        }

        if (btnConfirmSubmit) {
            btnConfirmSubmit.addEventListener("click", () => submitQuiz(false));
        }

        // Timer Countdown Implementation
        if (timeLimitMinutes > 0) {
            const timerDisplay = document.getElementById("timer-display");
            const timerBox = document.getElementById("quiz-timer-box");
            let remainingSeconds = timeLimitMinutes * 60;

            function formatTime(secs) {
                const m = Math.floor(secs / 60);
                const s = secs % 60;
                return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
            }

            if (timerDisplay) timerDisplay.textContent = formatTime(remainingSeconds);

            const timerInterval = setInterval(() => {
                if (isSubmitting) {
                    clearInterval(timerInterval);
                    return;
                }
                remainingSeconds--;
                if (timerDisplay) timerDisplay.textContent = formatTime(Math.max(0, remainingSeconds));

                if (remainingSeconds <= 60 && timerBox) {
                    timerBox.classList.add("timer-urgent");
                }

                if (remainingSeconds <= 0) {
                    clearInterval(timerInterval);
                    alert("Time is up! Your answers are being submitted automatically.");
                    submitQuiz(true);
                }
            }, 1000);
        }

        // Initialize First Question
        renderQuestion(0);
    }

    // -------------------------------------------------------------------------
    // Universal Responsive Mobile Navigation Drawer & Backdrop
    // -------------------------------------------------------------------------
    const sidebar = document.getElementById("appSidebar");
    if (sidebar) {
        // Ensure backdrop exists
        let backdrop = document.getElementById("sidebarBackdrop");
        if (!backdrop) {
            backdrop = document.createElement("div");
            backdrop.id = "sidebarBackdrop";
            backdrop.className = "sidebar-backdrop";
            document.body.appendChild(backdrop);
        }

        // Find or create toggle button in topbar
        let menuBtn = document.getElementById("mobileMenuBtn");
        const topbar = document.querySelector(".app-topbar, .editorial-topbar");
        if (!menuBtn && topbar) {
            menuBtn = document.createElement("button");
            menuBtn.id = "mobileMenuBtn";
            menuBtn.className = "mobile-sidebar-toggle";
            menuBtn.setAttribute("aria-label", "Toggle Navigation");
            menuBtn.innerHTML = "☰";
            topbar.prepend(menuBtn);
        }

        function toggleSidebar(e) {
            if (e) e.stopPropagation();
            const isOpen = sidebar.classList.toggle("mobile-open");
            if (backdrop) {
                backdrop.classList.toggle("active", isOpen);
            }
        }

        function closeSidebar() {
            sidebar.classList.remove("mobile-open");
            if (backdrop) backdrop.classList.remove("active");
        }

        if (menuBtn) {
            menuBtn.addEventListener("click", toggleSidebar);
        }

        if (backdrop) {
            backdrop.addEventListener("click", closeSidebar);
        }

        // Close drawer on escape key
        document.addEventListener("keydown", (e) => {
            if (e.key === "Escape" && sidebar.classList.contains("mobile-open")) {
                closeSidebar();
            }
        });
    }
});

