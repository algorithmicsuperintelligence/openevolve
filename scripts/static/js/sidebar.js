import { allNodeData, archiveProgramIds, formatMetrics, renderMetricBar, getHighlightNodes, selectedProgramId, setSelectedProgramId } from './main.js';
import { scrollAndSelectNodeById } from './graph.js';

const sidebar = document.getElementById('sidebar');
// Add a draggable resizer to let users change the sidebar width.
// Creates a slim handle at the left edge of the sidebar and uses pointer events
// to resize. The chosen width is persisted to localStorage under `sidebarWidth`.
(function enableSidebarResizer() {
    if (!sidebar) return;
    try {
        const STORAGE_KEY = 'sidebarWidth';
        const DEFAULT_WIDTH_PX = 360;
        const MIN_WIDTH_PX = 200;
        const MAX_WIDTH_PX = Math.max(window.innerWidth - 100, 400);

        // Restore saved width (if any)
        const saved = localStorage.getItem(STORAGE_KEY);
        if (saved) {
            sidebar.style.width = saved;
        } else if (!sidebar.style.width) {
            sidebar.style.width = DEFAULT_WIDTH_PX + 'px';
        }

        // Do not override sidebar positioning from CSS; assume #sidebar styles control placement

        // Create resizer element (left edge)
        const resizer = document.createElement('div');
        resizer.id = 'sidebar-resizer';
        resizer.setAttribute('role', 'separator');
        resizer.setAttribute('aria-orientation', 'vertical');
        resizer.setAttribute('tabindex', '0');
        // Make the hit area a bit larger and use flex to center an inner visible handle
        Object.assign(resizer.style, {
            position: 'fixed',
            left: '0px', // will be calculated
            top: '0px',
            width: '14px',
            cursor: 'col-resize',
            zIndex: '9999',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            background: 'transparent',
            transition: 'background 120ms',
            // disable pointerEvents by default so expanding sidebar doesn't immediately capture the mouse
            pointerEvents: 'none',
        });
        
        // Visible inner handle
        const handle = document.createElement('div');
        handle.id = 'sidebar-resizer-handle';
        handle.setAttribute('aria-hidden', 'true');
        Object.assign(handle.style, {
            width: '6px',
            height: '40px',
            borderRadius: '6px',
            // Use a subtle two-tone gradient and light border so it stands out in dark and light themes
            background: 'linear-gradient(180deg, rgba(255,255,255,0.9), rgba(200,200,200,0.6))',
            border: '1px solid rgba(0,0,0,0.12)',
            boxShadow: '0 1px 4px rgba(0,0,0,0.15)',
            transition: 'background 120ms, transform 120ms, box-shadow 120ms',
        });
        resizer.appendChild(handle);
        resizer.title = 'Drag to resize sidebar';

        // Hover/focus effects to make it obvious
        function _resizerHoverOn() { resizer.style.background = 'rgba(0,0,0,0.04)'; handle.style.transform = 'scale(1.06)'; handle.style.boxShadow = '0 2px 6px rgba(0,0,0,0.2)'; }
        function _resizerHoverOff() { resizer.style.background = 'transparent'; handle.style.transform = 'scale(1)'; handle.style.boxShadow = '0 1px 4px rgba(0,0,0,0.15)'; }
        resizer.addEventListener('pointerenter', _resizerHoverOn);
        resizer.addEventListener('pointerleave', _resizerHoverOff);
        resizer.addEventListener('focus', _resizerHoverOn);
        resizer.addEventListener('blur', _resizerHoverOff);

        // Insert the resizer as first child so it sits on the left edge
        // if (sidebar.firstChild) sidebar.insertBefore(resizer, sidebar.firstChild);
        // else sidebar.appendChild(resizer);
        // Append to body so it's not clipped by sidebar scrolling/overflow
        document.body.appendChild(resizer);

        // Position update function to align the fixed resizer with the sidebar left edge
        function updateResizerPosition() {
            const rect = sidebar.getBoundingClientRect();
            if (!rect || !isFinite(rect.left) || rect.width === 0) return;
            // Consider sidebar hidden if its left edge is at or past the right viewport edge
            const viewportRight = window.innerWidth || document.documentElement.clientWidth;
            const isOffscreen = rect.left >= (viewportRight - 8);
            if (isOffscreen || getComputedStyle(sidebar).display === 'none') {
                resizer.style.display = 'none';
                return;
            }
            // Ensure resizer is shown and aligned with the left edge of the sidebar
            resizer.style.display = 'flex';
            const left = Math.round(rect.left - 7);
            resizer.style.left = left + 'px';
            resizer.style.top = Math.round(rect.top) + 'px';
            resizer.style.height = Math.max(40, Math.round(rect.height)) + 'px';
        }
        // Initial position
        updateResizerPosition();
        // Keep in sync on resize and mutation of sidebar attributes
        window.addEventListener('resize', updateResizerPosition);
        const mo = new MutationObserver(updateResizerPosition);
        mo.observe(sidebar, { attributes: true, attributeFilter: ['style', 'class'] });

        // Continuous updating while sidebar transitions or when mouse moves near the edge
        let rafId = null;
        function rafLoop() {
            updateResizerPosition();
            rafId = requestAnimationFrame(rafLoop);
        }
        function startContinuousUpdate() {
            if (!rafId) rafLoop();
        }
        function stopContinuousUpdate() {
            if (rafId) {
                cancelAnimationFrame(rafId);
                rafId = null;
            }
        }

        // If the sidebar has a CSS transition on transform, run rAF during it to keep alignment
        sidebar.addEventListener('transitionstart', startContinuousUpdate);
        sidebar.addEventListener('transitionend', function() { updateResizerPosition(); stopContinuousUpdate(); });
        sidebar.addEventListener('transitioncancel', function() { updateResizerPosition(); stopContinuousUpdate(); });

        // Track last mouse position and proximity-based enabling of pointer events
        // to avoid the sidebar itself stealing the pointer when it expands under the cursor.
        const PROXIMITY_PX = 28;
        let mousePending = false;
        let lastMouseX = null;
        let lastMouseY = null;
        function checkPointerProximity(clientX, clientY) {
            if (!resizer || resizer.style.display === 'none') return;
            const r = resizer.getBoundingClientRect();
            if (!r || r.width === 0) return;
            // Compute distance to the resizer vertical centerline
            const dx = Math.max(r.left - clientX, clientX - (r.left + r.width));
            const dy = Math.max(r.top - clientY, clientY - (r.top + r.height));
            const within = (dx <= PROXIMITY_PX && dy <= PROXIMITY_PX) || (clientX >= r.left && clientX <= r.left + r.width && clientY >= r.top && clientY <= r.top + r.height);
            if (within) {
                if (resizer.style.pointerEvents !== 'auto') {
                    resizer.style.pointerEvents = 'auto';
                    resizer.classList.add('resizer-proximate');
                }
            } else {
                if (resizer.style.pointerEvents !== 'none' && !isResizing) {
                    resizer.style.pointerEvents = 'none';
                    resizer.classList.remove('resizer-proximate');
                }
            }
            // Also control the sidebar's pointer events: only enable when within proximity, when resizing, or when sidebar is sticky
            try {
                const srect = sidebar.getBoundingClientRect();
                const viewportRight = window.innerWidth || document.documentElement.clientWidth;
                const isOffscreen = srect.left >= (viewportRight - 8);
                if (isOffscreen || getComputedStyle(sidebar).display === 'none') {
                    sidebar.style.pointerEvents = 'none';
                } else if (within || isResizing || sidebarSticky) {
                    sidebar.style.pointerEvents = 'auto';
                } else {
                    // keep the sidebar unclickable unless cursor is near it
                    sidebar.style.pointerEvents = 'none';
                }
            } catch (err) {
                // ignore
            }
        }
        document.addEventListener('mousemove', function (e) {
            lastMouseX = e.clientX; lastMouseY = e.clientY;
            if (!mousePending) {
                mousePending = true;
                requestAnimationFrame(function () {
                    updateResizerPosition();
                    checkPointerProximity(lastMouseX, lastMouseY);
                    mousePending = false;
                });
            }
        });

        let isResizing = false;
        let startX = 0;
        let startWidth = 0;

        function clampWidth(w) {
            const max = Math.min(MAX_WIDTH_PX, Math.floor(window.innerWidth - 100));
            return Math.max(MIN_WIDTH_PX, Math.min(w, max));
        }

        function onPointerMove(e) {
            if (!isResizing) return;
            const dx = e.clientX - startX; // positive when moving right
            // Since the resizer is on the left edge of a right-aligned sidebar,
            // moving pointer to the right should make the sidebar narrower.
            // Compute new width as startWidth - dx.
            let newWidth = Math.round(startWidth - dx);
            newWidth = clampWidth(newWidth);
            sidebar.style.width = newWidth + 'px';
        }

        function onPointerUp(e) {
            if (!isResizing) return;
            isResizing = false;
            document.body.style.cursor = '';
            document.body.style.userSelect = '';
            try { localStorage.setItem(STORAGE_KEY, sidebar.style.width); } catch (err) { /* ignore */ }
            // Remove global listeners
            document.removeEventListener('pointermove', onPointerMove);
            document.removeEventListener('pointerup', onPointerUp);
        }

        resizer.addEventListener('pointerdown', (e) => {
            e.preventDefault();
            isResizing = true;
            startX = e.clientX;
            startWidth = parseInt(window.getComputedStyle(sidebar).width, 10) || DEFAULT_WIDTH_PX;
            document.body.style.cursor = 'col-resize';
            document.body.style.userSelect = 'none';
            document.addEventListener('pointermove', onPointerMove);
            document.addEventListener('pointerup', onPointerUp);
            // attempt to capture pointer so touch works well
            try { e.target.setPointerCapture && e.target.setPointerCapture(e.pointerId); } catch (err) { }
            // Update resizer position during drag (useful when width changes)
            updateResizerPosition();
        });

        // Keyboard accessibility: left/right arrows adjust width
        resizer.addEventListener('keydown', (e) => {
            const step = 20;
            let cur = parseInt(window.getComputedStyle(sidebar).width, 10) || DEFAULT_WIDTH_PX;
            if (e.key === 'ArrowLeft') {
                cur = clampWidth(cur - step);
                sidebar.style.width = cur + 'px';
                try { localStorage.setItem(STORAGE_KEY, sidebar.style.width); } catch (err) {}
                e.preventDefault();
            } else if (e.key === 'ArrowRight') {
                cur = clampWidth(cur + step);
                sidebar.style.width = cur + 'px';
                try { localStorage.setItem(STORAGE_KEY, sidebar.style.width); } catch (err) {}
                e.preventDefault();
            } else if (e.key === 'Home') {
                sidebar.style.width = DEFAULT_WIDTH_PX + 'px';
                try { localStorage.setItem(STORAGE_KEY, sidebar.style.width); } catch (err) {}
                e.preventDefault();
            }
        });

        // Make sure the stored max width updates on window resize
        window.addEventListener('resize', () => {
            const cur = parseInt(window.getComputedStyle(sidebar).width, 10) || DEFAULT_WIDTH_PX;
            const clamped = clampWidth(cur);
            if (clamped !== cur) {
                sidebar.style.width = clamped + 'px';
                try { localStorage.setItem(STORAGE_KEY, sidebar.style.width); } catch (err) {}
            }
            updateResizerPosition();
        });
        // When sidebar is shown/hidden via showSidebar/hideSidebar functions, keep resizer sync
        const showHideObserver = new MutationObserver(updateResizerPosition);
        showHideObserver.observe(sidebar, { attributes: true, attributeFilter: ['style', 'class'] });
    } catch (err) {
        // don't crash the rest of the sidebar code if resizing support fails
        console.warn('sidebar resizer init failed', err);
    }
})();

export let sidebarSticky = false;
let lastSidebarTab = null;

export function showSidebar() {
    sidebar.style.transform = 'translateX(0)';
    // When explicitly shown, enable pointer events so controls are interactive
    try { sidebar.style.pointerEvents = 'auto'; } catch (e) {}
}
export function hideSidebar() {
    sidebar.style.transform = 'translateX(100%)';
    sidebarSticky = false;
    try { sidebar.style.pointerEvents = 'none'; } catch (e) {}
}

export function showSidebarContent(d, fromHover = false) {
    const sidebarContent = document.getElementById('sidebar-content');
    if (!sidebarContent) return;
    if (fromHover && sidebarSticky) return;
    if (!d) {
        sidebarContent.innerHTML = '';
        return;
    }
    let starHtml = '';
    if (archiveProgramIds && archiveProgramIds.includes(d.id)) {
        starHtml = '<span style="position:relative;top:0.05em;left:0.15em;font-size:1.6em;color:#FFD600;z-index:10;" title="MAP-elites member" aria-label="MAP-elites member">★</span>';
    }
    let locatorBtn = '<button id="sidebar-locator-btn" title="Locate selected node" aria-label="Locate selected node" style="position:absolute;top:0.05em;right:2.5em;font-size:1.5em;background:none;border:none;color:#FFD600;cursor:pointer;z-index:10;line-height:1;filter:drop-shadow(0 0 2px #FFD600);">⦿</button>';
    let closeBtn = '<button id="sidebar-close-btn" style="position:absolute;top:0.05em;right:0.15em;font-size:1.6em;background:none;border:none;color:#888;cursor:pointer;z-index:10;line-height:1;">&times;</button>';
    let openLink = '<div style="text-align:center;margin:-1em 0 1.2em 0;"><a href="/program/' + d.id + '" target="_blank" class="open-in-new" style="font-size:0.95em;">[open in new window]</a></div>';
    let tabHtml = '';
    let tabContentHtml = '';
    let tabNames = [];
    if (d.code && typeof d.code === 'string' && d.code.trim() !== '') tabNames.push('Code');
    if ((d.prompts && typeof d.prompts === 'object' && Object.keys(d.prompts).length > 0) || (d.artifacts_json && typeof d.artifacts_json === 'object' && Object.keys(d.artifacts_json).length > 0)) tabNames.push('Prompts');
    const children = allNodeData.filter(n => n.parent_id === d.id);
    if (children.length > 0) tabNames.push('Children');

    // Handle nodes with "-copyN" IDs
    function getBaseId(id) {
        return id.includes('-copy') ? id.split('-copy')[0] : id;
    }
    const baseId = getBaseId(d.id);
    const clones = allNodeData.filter(n => getBaseId(n.id) === baseId && n.id !== d.id);
    if (clones.length > 0) tabNames.push('Clones');

    // Add a Diff tab when a parent exists with code to compare against
    const parentNodeForDiff = d.parent_id && d.parent_id !== 'None' ? allNodeData.find(n => n.id == d.parent_id) : null;
    if (parentNodeForDiff && parentNodeForDiff.code && parentNodeForDiff.code.trim() !== '') {
        tabNames.push('Diff');
    }

    // Add a Repairs tab when LLM repair was performed on this program
    if (d.metadata && d.metadata.repair_history && d.metadata.repair_history.length > 0) {
        tabNames.push('Repairs');
    }

        let activeTab = lastSidebarTab && tabNames.includes(lastSidebarTab) ? lastSidebarTab : tabNames[0];
 
        // Helper to render tab content
        // Simple line-level LCS diff renderer between two code strings
        function renderCodeDiff(aCode, bCode) {
            const a = (aCode || '').split('\n');
            const b = (bCode || '').split('\n');
            const m = a.length, n = b.length;
            // build LCS table
            const dp = Array.from({length: m+1}, () => new Array(n+1).fill(0));
            for (let ii = m-1; ii >= 0; --ii) {
                for (let jj = n-1; jj >= 0; --jj) {
                    if (a[ii] === b[jj]) dp[ii][jj] = dp[ii+1][jj+1] + 1;
                    else dp[ii][jj] = Math.max(dp[ii+1][jj], dp[ii][jj+1]);
                }
            }
            // backtrack
            let i = 0, j = 0;
            const parts = [];
            while (i < m && j < n) {
                if (a[i] === b[j]) {
                    parts.push({type: 'eq', line: a[i]});
                    i++; j++; 
                } else if (dp[i+1][j] >= dp[i][j+1]) {
                    parts.push({type: 'del', line: a[i]});
                    i++;
                } else {
                    parts.push({type: 'ins', line: b[j]});
                    j++;
                }
            }
            while (i < m) { parts.push({type: 'del', line: a[i++]}); }
            while (j < n) { parts.push({type: 'ins', line: b[j++]}); }

            // Render HTML with inline styles
            const htmlLines = parts.map(function(p) {
                if (p.type === 'eq') return '<div style="white-space:pre-wrap;">' + escapeHtml(p.line) + '</div>';
                if (p.type === 'del') return '<div style="background:#fff0f0;color:#8b1a1a;padding:0.08em 0.3em;border-left:3px solid #f26;white-space:pre-wrap;">- ' + escapeHtml(p.line) + '</div>';
                return '<div style="background:#f2fff2;color:#116611;padding:0.08em 0.3em;border-left:3px solid #2a8;white-space:pre-wrap;">+ ' + escapeHtml(p.line) + '</div>';
            });
            return '<div style="font-family: \'Fira Mono\', monospace; font-size:0.95em; line-height:1.35;">' +
                '<div style="margin-bottom:0.4em;color:#666;">Showing diff between program and its parent (parent id: ' + (parentNodeForDiff ? parentNodeForDiff.id : 'N/A') + ')</div>' +
                htmlLines.join('') + '</div>';
        }

        // small helper to escape HTML
        function escapeHtml(s) {
            return (s+'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
        }

        function renderSidebarTabContent(tabName, d, children) {
            if (tabName === 'Code') {
                return `<pre class="sidebar-code-pre">${escapeHtml(d.code)}</pre>`;
            }
            if (tabName === 'Prompts') {
                // Prompt select logic
                let promptOptions = [];
                let promptMap = {};
                if (d.prompts && typeof d.prompts === 'object') {
                    for (const [k, v] of Object.entries(d.prompts)) {
                        if (v && typeof v === 'object' && !Array.isArray(v)) {
                            for (const [subKey, subVal] of Object.entries(v)) {
                                 const optLabel = `${k} - ${subKey}`;
                                 promptOptions.push(optLabel);
                                 promptMap[optLabel] = subVal;
                             }
                         } else {
                             const optLabel = `${k}`;
                             promptOptions.push(optLabel);
                             promptMap[optLabel] = v;
                         }
                     }
                 }
                 // Artifacts
                 if (d.artifacts_json) {
                     const optLabel = `artifacts`;
                     promptOptions.push(optLabel);
                     promptMap[optLabel] = d.artifacts_json;
                 }
                 // Get last selected prompt from localStorage, or default to first
                 let lastPromptKey = localStorage.getItem('sidebarPromptSelect') || promptOptions[0] || '';
                 if (!promptMap[lastPromptKey]) lastPromptKey = promptOptions[0] || '';
                 // Build select box
                 let selectHtml = '';
                 if (promptOptions.length > 1) {
                     selectHtml = `<select id="sidebar-prompt-select" style="margin-bottom:0.7em;max-width:100%;font-size:1em;">
                         ${promptOptions.map(opt => `<option value="${opt}"${opt===lastPromptKey?' selected':''}>${opt}</option>`).join('')}
                     </select>`;
                 }
                 // Show only the selected prompt
                 let promptVal = promptMap[lastPromptKey];
                 
                 // Handle unicode escape for artifacts JSON display
                 if (lastPromptKey === 'artifacts' && typeof promptVal === 'string') {
                     try {
                         // Parse and stringify to properly escape unicode
                         const parsed = JSON.parse(promptVal);
                         promptVal = JSON.stringify(parsed, null, 2);
                     } catch (e) {
                         // If parsing fails, use original value
                         console.warn('Failed to parse artifacts JSON for unicode escape:', e);
                     }
                 }

                 let promptHtml = `<pre class="sidebar-pre">${promptVal ?? ''}</pre>`;
                 return selectHtml + promptHtml;
             }
             if (tabName === 'Children') {
                 const metric = (document.getElementById('metric-select') && document.getElementById('metric-select').value) || 'combined_score';
                 let min = 0, max = 1;
                 const vals = children.map(child => (child.metrics && typeof child.metrics[metric] === 'number') ? child.metrics[metric] : null).filter(x => x !== null);
                 if (vals.length > 0) {
                     min = Math.min(...vals);
                     max = Math.max(...vals);
                 }
                 return `<div><ul style='margin:0.5em 0 0 1em;padding:0;'>` +
                     children.map(child => {
                         let val = (child.metrics && typeof child.metrics[metric] === 'number') ? child.metrics[metric].toFixed(4) : '(no value)';
                         let bar = (child.metrics && typeof child.metrics[metric] === 'number') ? renderMetricBar(child.metrics[metric], min, max) : '';
                         return `<li style='margin-bottom:0.3em;'><a href="#" class="child-link" data-child="${child.id}">${child.id}</a><br /><br /> <span style='margin-left:0.5em;'>${val}</span> ${bar}</li>`;
                     }).join('') +
                     `</ul></div>`;
             }
             if (tabName === 'Clones') {
                 return `<div><ul style='margin:0.5em 0 0 1em;padding:0;'>` +
                     clones.map(clone =>
                         `<li style='margin-bottom:0.3em;'><a href="#" class="clone-link" data-clone="${clone.id}">${clone.id}</a></li>`
                     ).join('') +
                     `</ul></div>`;
             }
             if (tabName === 'Diff') {
                 const parentNode = parentNodeForDiff;
                 const parentCode = parentNode ? parentNode.code || '' : '';
                 const curCode = d.code || '';
                 return renderCodeDiff(parentCode, curCode);
             }
             if (tabName === 'Repairs') {
                 const originalCode = (d.metadata && d.metadata.original_llm_code) || '';
                 const repairedCode = d.code || '';
                 const history = (d.metadata && d.metadata.repair_history) || [];
                 // Attempt table
                 const tableRows = history.map(function(entry) {
                     const badge = entry.succeeded
                         ? '<span style="display:inline-block;padding:1px 7px;border-radius:3px;background:#1a7f3c;color:#fff;font-weight:600;font-size:0.88em;">✓ ok</span>'
                         : '<span style="display:inline-block;padding:1px 7px;border-radius:3px;background:#b91c1c;color:#fff;font-weight:600;font-size:0.88em;">✗ fail</span>';
                     const errText = escapeHtml(entry.error || entry.repair_error || '—');
                     return '<tr style="border-bottom:1px solid #ddd;">' +
                         '<td style="padding:5px 10px;text-align:center;font-weight:600;color:#444;">' + (entry.attempt || '') + '</td>' +
                         '<td style="padding:5px 10px;">' + badge + '</td>' +
                         '<td style="padding:5px 10px;font-size:0.83em;white-space:pre-wrap;word-break:break-all;color:#333;font-family:monospace;">' + errText + '</td>' +
                         '</tr>';
                 }).join('');
                 const tableHtml = '<table style="border-collapse:collapse;width:100%;margin-bottom:0.9em;font-size:0.91em;background:#fff;border:1px solid #ddd;border-radius:4px;overflow:hidden;">' +
                     '<thead><tr style="background:#2d3748;color:#fff;">' +
                     '<th style="padding:5px 10px;font-weight:600;text-align:center;width:70px;">#</th>' +
                     '<th style="padding:5px 10px;font-weight:600;width:90px;">Status</th>' +
                     '<th style="padding:5px 10px;font-weight:600;">Compiler output / notes</th>' +
                     '</tr></thead><tbody>' + tableRows + '</tbody></table>';
                 const diffHtml = originalCode
                     ? '<div style="margin-bottom:0.3em;color:#666;font-size:0.9em;">Diff: original LLM output → repaired code</div>' +
                       renderCodeDiff(originalCode, repairedCode)
                     : '<div style="color:#888;font-size:0.9em;">(Original code not recorded)</div>';
                 return tableHtml + diffHtml;
             }
             return '';
         }
 
     if (tabNames.length > 0) {
         tabHtml = '<div id="sidebar-tab-bar" style="display:flex;gap:0.7em;margin-bottom:0.7em;">' +
             tabNames.map((name) => `<span class="sidebar-tab${name===activeTab?' active':''}" data-tab="${name}">${name}</span>`).join('') + '</div>';
         tabContentHtml = `<div id="sidebar-tab-content">${renderSidebarTabContent(activeTab, d, children)}</div>`;
     }
    let parentIslandHtml = '';
    if (d.parent_id && d.parent_id !== 'None') {
        const parent = allNodeData.find(n => n.id == d.parent_id);
        if (parent && parent.island !== undefined) {
            parentIslandHtml = ` <span style="color:#888;font-size:0.92em;">(island ${parent.island})</span>`;
        }
    }
    sidebarContent.innerHTML =
        `<div style="position:relative;min-height:2em;">
            ${starHtml}
            ${locatorBtn}
            ${closeBtn}
            ${openLink}
            <b>Program ID:</b> ${d.id}<br>
            <b>Island:</b> ${d.island}<br>
            <b>Generation:</b> ${d.generation}<br>
            <b>Parent ID:</b> <a href="#" class="parent-link" data-parent="${d.parent_id || ''}">${d.parent_id || 'None'}</a>${parentIslandHtml}<br><br>
            <b>Metrics:</b><br>${formatMetrics(d.metrics)}<br><br>
            ${tabHtml}${tabContentHtml}
        </div>`;

    // Helper to attach prompt select handler
    function attachPromptSelectHandler() {
        const promptSelect = document.getElementById('sidebar-prompt-select');
        if (promptSelect) {
            promptSelect.onchange = function() {
                localStorage.setItem('sidebarPromptSelect', promptSelect.value);
                // Only re-render the Prompts tab, not the whole sidebar
                const tabContent = document.getElementById('sidebar-tab-content');
                if (tabContent) {
                    tabContent.innerHTML = renderSidebarTabContent('Prompts', d, children);
                    attachPromptSelectHandler();
                }
            };
        }
    }
    attachPromptSelectHandler();

    if (tabNames.length > 1) {
        const tabBar = document.getElementById('sidebar-tab-bar');
        Array.from(tabBar.children).forEach(tabEl => {
            tabEl.onclick = function() {
                Array.from(tabBar.children).forEach(e => e.classList.remove('active'));
                tabEl.classList.add('active');
                const tabName = tabEl.dataset.tab;
                lastSidebarTab = tabName;
                const tabContent = document.getElementById('sidebar-tab-content');
                tabContent.innerHTML = renderSidebarTabContent(tabName, d, children);
                if (tabName === 'Prompts') {
                    attachPromptSelectHandler();
                }
                setTimeout(() => {
                    document.querySelectorAll('.child-link').forEach(link => {
                        link.onclick = function(e) {
                            e.preventDefault();
                            const childNode = allNodeData.find(n => n.id == link.dataset.child);
                            if (childNode) {
                                window._lastSelectedNodeData = childNode;
                                const perfTabBtn = document.getElementById('tab-performance');
                                const perfTabView = document.getElementById('view-performance');
                                if ((perfTabBtn && perfTabBtn.classList.contains('active')) || (perfTabView && perfTabView.classList.contains('active'))) {
                                    import('./performance.js').then(mod => {
                                        mod.selectPerformanceNodeById(childNode.id);
                                        showSidebar();
                                    });
                                } else {
                                    scrollAndSelectNodeById(childNode.id);
                                }
                            }
                        };
                    });
                    document.querySelectorAll('.clone-link').forEach(link => {
                        link.onclick = function(e) {
                            e.preventDefault();
                            const cloneNode = allNodeData.find(n => n.id == link.dataset.clone);
                            if (cloneNode) {
                                window._lastSelectedNodeData = cloneNode;
                                const perfTabBtn = document.getElementById('tab-performance');
                                const perfTabView = document.getElementById('view-performance');
                                if ((perfTabBtn && perfTabBtn.classList.contains('active')) || (perfTabView && perfTabView.classList.contains('active'))) {
                                    import('./performance.js').then(mod => {
                                        mod.selectPerformanceNodeById(cloneNode.id);
                                        showSidebar();
                                    });
                                } else {
                                    scrollAndSelectNodeById(cloneNode.id);
                                }
                            }
                        };
                    });
                }, 0);
            };
        });
    }
    setTimeout(() => {
        attachPromptSelectHandler();
        document.querySelectorAll('.child-link').forEach(link => {
            link.onclick = function(e) {
                e.preventDefault();
                const childNode = allNodeData.find(n => n.id == link.dataset.child);
                if (childNode) {
                    window._lastSelectedNodeData = childNode;
                    // Check if performance tab is active
                    const perfTabBtn = document.getElementById('tab-performance');
                    const perfTabView = document.getElementById('view-performance');
                    if ((perfTabBtn && perfTabBtn.classList.contains('active')) || (perfTabView && perfTabView.classList.contains('active'))) {
                        import('./performance.js').then(mod => {
                            mod.selectPerformanceNodeById(childNode.id);
                            showSidebar();
                        });
                    } else {
                        scrollAndSelectNodeById(childNode.id);
                    }
                }
            };
        });
        document.querySelectorAll('.clone-link').forEach(link => {
            link.onclick = function(e) {
                e.preventDefault();
                const cloneNode = allNodeData.find(n => n.id == link.dataset.clone);
                if (cloneNode) {
                    window._lastSelectedNodeData = cloneNode;
                    const perfTabBtn = document.getElementById('tab-performance');
                    const perfTabView = document.getElementById('view-performance');
                    if ((perfTabBtn && perfTabBtn.classList.contains('active')) || (perfTabView && perfTabView.classList.contains('active'))) {
                        import('./performance.js').then(mod => {
                            mod.selectPerformanceNodeById(cloneNode.id);
                            showSidebar();
                        });
                    } else {
                        scrollAndSelectNodeById(cloneNode.id);
                    }
                }
            };
        });
    }, 0);
    const closeBtnEl = document.getElementById('sidebar-close-btn');
    if (closeBtnEl) closeBtnEl.onclick = function() {
        setSelectedProgramId(null);
        sidebarSticky = false;
        hideSidebar();
    };
    // Locator button logic
    const locatorBtnEl = document.getElementById('sidebar-locator-btn');
    if (locatorBtnEl) {
        locatorBtnEl.onclick = function(e) {
            e.preventDefault();
            // Use view display property for active view detection
            const viewBranching = document.getElementById('view-branching');
            const viewPerformance = document.getElementById('view-performance');
            const viewList = document.getElementById('view-list');
            if (viewBranching && viewBranching.style.display !== 'none') {
                import('./graph.js').then(mod => {
                    mod.centerAndHighlightNodeInGraph(d.id);
                });
            } else if (viewPerformance && viewPerformance.style.display !== 'none') {
                import('./performance.js').then(mod => {
                    mod.centerAndHighlightNodeInPerformanceGraph(d.id);
                });
            } else if (viewList && viewList.style.display !== 'none') {
                // Scroll to list item
                const container = document.getElementById('node-list-container');
                if (container) {
                    const rows = Array.from(container.children);
                    const target = rows.find(div => div.getAttribute('data-node-id') === d.id);
                    if (target) {
                        target.scrollIntoView({behavior: 'smooth', block: 'center'});
                        // Optionally add a yellow highlight effect
                        target.classList.add('node-locator-highlight');
                        setTimeout(() => target.classList.remove('node-locator-highlight'), 1000);
                    }
                }
            }
        };
    }
    // Parent link logic
    const parentLink = sidebarContent.querySelector('.parent-link');
    if (parentLink && parentLink.dataset.parent && parentLink.dataset.parent !== 'None' && parentLink.dataset.parent !== '') {
        parentLink.onclick = function(e) {
            e.preventDefault();
            const parentNode = allNodeData.find(n => n.id == parentLink.dataset.parent);
            if (parentNode) {
                window._lastSelectedNodeData = parentNode;
            }
            const perfTabBtn = document.getElementById('tab-performance');
            const perfTabView = document.getElementById('view-performance');
            if ((perfTabBtn && perfTabBtn.classList.contains('active')) || (perfTabView && perfTabView.classList.contains('active'))) {
                import('./performance.js').then(mod => {
                    mod.selectPerformanceNodeById(parentLink.dataset.parent);
                    showSidebar();
                });
            } else {
                scrollAndSelectNodeById(parentLink.dataset.parent);
            }
        };
    }
}

export function openInNewTab(event, d) {
    const url = `/program/${d.id}`;
    window.open(url, '_blank');
    event.stopPropagation();
}

export function setSidebarSticky(val) {
    sidebarSticky = val;
    try {
        sidebar.style.pointerEvents = val ? 'auto' : 'none';
    } catch (e) {}
}
// Helper to escape HTML so code can be shown verbatim inside <pre>
function escapeHtml(str) {
    if (str === undefined || str === null) return '';
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}