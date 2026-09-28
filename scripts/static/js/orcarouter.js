/*
 * OrcaRouter provider settings page.
 *
 * Two credential choices are presented side by side and both go through the
 * same local server, which owns the key. The browser never receives a secret.
 *
 * Asynchronous login work is guarded by a monotonically increasing generation.
 * `pagehide` cannot rely on the guarded `finally` block: it clears the busy
 * flag and hint synchronously and then asks the server to cancel with
 * `keepalive`, so a back-forward-cache restore is not stuck busy and a second
 * login can start without remounting.
 */

(function () {
  "use strict";

  var state = {
    generation: 0,
    attemptId: null,
    pollTimer: null,
    loginBusy: false,
    selectedModel: null,
    options: [],
    source: null,
    dropdownOpen: false,
  };

  function $(id) {
    return document.getElementById(id);
  }

  function setStatus(el, text, tone) {
    if (!el) return;
    el.textContent = text || "";
    if (tone) {
      el.setAttribute("data-tone", tone);
    } else {
      el.removeAttribute("data-tone");
    }
  }

  function request(path, options) {
    return fetch(path, options).then(function (response) {
      return response.json().catch(function () {
        return {};
      }).then(function (body) {
        return { ok: response.ok, status: response.status, body: body };
      });
    });
  }

  // ---------------------------------------------------------------- status

  function refreshStatus() {
    return request("/orcarouter/api/status").then(function (result) {
      var s = result.body || {};
      $("stAuthenticated").textContent = s.authenticated ? "yes" : "no";
      $("stSecret").textContent = s.secret_masked || "—";
      $("stSource").textContent = s.auth_source || "—";
      $("stGeneration").textContent =
        s.generation === null || s.generation === undefined ? "—" : String(s.generation);
      $("stScope").textContent = s.scope || "—";
      $("stApiBase").textContent = s.api_base || "—";
      if (s.needs_reauth) {
        setStatus(
          $("oauthStatus"),
          "This credential was rejected. Sign in again or paste a new key.",
          "error"
        );
      }
      return s;
    });
  }

  // ------------------------------------------------------------- API key

  function saveApiKey() {
    var input = $("apiKeyInput");
    var key = (input.value || "").trim();
    setStatus($("apiKeyStatus"), "Saving…");
    request("/orcarouter/api/key", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ api_key: key }),
    }).then(function (result) {
      if (!result.ok) {
        setStatus($("apiKeyStatus"), result.body.error || "Could not save that key.", "error");
        return;
      }
      input.value = "";
      setStatus($("apiKeyStatus"), "Stored " + result.body.secret_masked, "ok");
      refreshStatus().then(loadModels);
    });
  }

  function clearApiKey() {
    request("/orcarouter/api/key/clear", { method: "POST" }).then(function () {
      setStatus($("apiKeyStatus"), "Credential cleared.", "ok");
      refreshStatus().then(loadModels);
    });
  }

  // ---------------------------------------------------------------- login

  function stopPolling() {
    if (state.pollTimer) {
      clearInterval(state.pollTimer);
      state.pollTimer = null;
    }
  }

  function setLoginBusy(busy, hint) {
    state.loginBusy = busy;
    $("connectBtn").disabled = busy;
    $("cancelLoginBtn").disabled = !busy;
    $("connectBtn").setAttribute("aria-busy", busy ? "true" : "false");
    if (hint !== undefined) {
      var urlEl = $("authorizeUrl");
      if (hint) {
        urlEl.textContent = hint;
        urlEl.hidden = false;
      } else {
        urlEl.textContent = "";
        urlEl.hidden = true;
      }
    }
  }

  function startLogin() {
    stopPolling();
    state.generation += 1;
    var generation = state.generation;
    setLoginBusy(true, "");
    setStatus($("oauthStatus"), "Starting sign-in…");

    request("/orcarouter/api/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ oob: $("oobToggle").checked }),
    }).then(function (result) {
      if (generation !== state.generation) return;
      state.attemptId = result.body.attempt_id;
      state.pollTimer = setInterval(function () {
        pollLogin(generation, state.attemptId);
      }, 1000);
      pollLogin(generation, state.attemptId);
    });
  }

  function pollLogin(generation, attemptId) {
    if (generation !== state.generation) return;
    request("/orcarouter/api/login/" + encodeURIComponent(attemptId)).then(function (result) {
      if (generation !== state.generation) return;
      var body = result.body || {};
      if (body.url) {
        setLoginBusy(true, body.url);
      }
      if (body.state === "pending") {
        setStatus($("oauthStatus"), "Waiting for you to approve in the browser…");
        return;
      }
      stopPolling();
      setLoginBusy(false, "");
      if (body.state === "success") {
        setStatus($("oauthStatus"), "Signed in. Stored " + body.secret_masked, "ok");
        refreshStatus().then(loadModels);
      } else if (body.state === "error") {
        setStatus($("oauthStatus"), body.error || "Sign-in failed.", "error");
      } else {
        setStatus($("oauthStatus"), "Sign-in cancelled.", "warn");
      }
    });
  }

  function cancelLogin(keepalive) {
    stopPolling();
    state.generation += 1;
    setLoginBusy(false, "");
    setStatus($("oauthStatus"), "Sign-in cancelled.", "warn");
    var attemptId = state.attemptId;
    state.attemptId = null;
    if (!attemptId) return;
    fetch("/orcarouter/api/login/" + encodeURIComponent(attemptId) + "/cancel", {
      method: "POST",
      keepalive: !!keepalive,
    }).catch(function () {
      /* the generation guard already dropped the local state */
    });
  }

  // -------------------------------------------------------------- catalog

  function loadModels() {
    var capability = $("capabilitySelect").value;
    var modality = $("modalitySelect").value;
    setStatus($("modelStatus"), "Loading models…");
    return request(
      "/orcarouter/api/models?capability=" +
        encodeURIComponent(capability) +
        "&modality=" +
        encodeURIComponent(modality)
    ).then(function (result) {
      var body = result.body || {};
      state.options = body.models || [];
      state.source = body.source || null;
      renderOptions();
      var tone = "ok";
      var parts = [state.options.length + " models"];
      if (body.source) parts.push("source: " + body.source);
      if (body.degraded) {
        tone = "warn";
        parts.push("degraded — verified fallback catalog");
        if (body.error) parts.push(body.error);
      }
      if (!body.authenticated) {
        tone = "warn";
        parts.push("not signed in — this is the public catalog, not your workspace");
      }
      if (body.catalog_source_url) {
        parts.push(body.catalog_source_url);
      }
      if (body.needs_reauth) {
        tone = "error";
        parts.push("credential rejected — sign in again or paste a new key");
      }
      if (!state.options.length) {
        tone = body.degraded ? "warn" : "error";
        parts.push("no compatible models");
      }
      setStatus($("modelStatus"), parts.join(" · "), tone);
      ensureSelectionIsCompatible();
    });
  }

  function renderOptions() {
    var panel = $("modelPanel");
    panel.textContent = "";
    if (!state.options.length) {
      var empty = document.createElement("div");
      empty.className = "option";
      empty.textContent = "No compatible models";
      panel.appendChild(empty);
      return;
    }
    state.options.forEach(function (option) {
      var row = document.createElement("div");
      row.className = "option";
      row.setAttribute("role", "option");
      row.setAttribute("data-model-id", option.id);
      row.setAttribute("data-catalog-source", state.source || "");
      row.setAttribute("aria-selected", option.id === state.selectedModel ? "true" : "false");

      var name = document.createElement("span");
      name.className = "name";
      name.textContent = option.label || option.id;
      row.appendChild(name);

      var meta = document.createElement("span");
      meta.className = "meta";
      var bits = [];
      if (option.context_length) bits.push(Math.round(option.context_length / 1000) + "k ctx");
      var modalities = (option.input_modalities || []).join("/") || "text";
      bits.push(modalities);
      if (option.reasoning_efforts && option.reasoning_efforts.length) {
        bits.push(option.reasoning_efforts.join("/"));
      }
      meta.textContent = bits.join(" · ");
      row.appendChild(meta);

      if (option.verified) {
        var tag = document.createElement("span");
        tag.className = "fallback-tag";
        tag.textContent = "verified fallback";
        row.appendChild(tag);
      }

      row.addEventListener("click", function () {
        selectModel(option);
      });
      panel.appendChild(row);
    });
  }

  function selectModel(option) {
    state.selectedModel = option.id;
    $("modelTriggerLabel").textContent = option.label || option.id;
    closeDropdown();
    var list = $("selectedModel");
    list.textContent = "";
    var item = document.createElement("li");
    var code = document.createElement("code");
    code.textContent = option.id;
    item.appendChild(code);
    list.appendChild(item);
    list.hidden = false;
    renderOptions();
  }

  function ensureSelectionIsCompatible() {
    if (!state.selectedModel) return;
    var stillThere = state.options.some(function (option) {
      return option.id === state.selectedModel;
    });
    if (!stillThere) {
      var dropped = state.selectedModel;
      state.selectedModel = null;
      $("modelTriggerLabel").textContent = "Select a model…";
      $("selectedModel").hidden = true;
      setStatus(
        $("modelStatus"),
        "Removed '" + dropped + "': it is not offered for the current capability/modality. Pick another model.",
        "warn"
      );
    }
  }

  function openDropdown() {
    state.dropdownOpen = true;
    $("modelPanel").hidden = false;
    $("modelTrigger").setAttribute("aria-expanded", "true");
  }

  function closeDropdown() {
    state.dropdownOpen = false;
    $("modelPanel").hidden = true;
    $("modelTrigger").setAttribute("aria-expanded", "false");
  }

  // ------------------------------------------------------------ lifecycle

  function onPageHide() {
    // Back-forward cache: clear busy/hint synchronously, invalidate the
    // generation, and cancel server-side work with keepalive. The guarded
    // `finally` of the invalidated poll would refuse to mutate state, which
    // would leave a restored page permanently busy.
    stopPolling();
    state.generation += 1;
    state.loginBusy = false;
    setLoginBusy(false, "");
    setStatus($("oauthStatus"), "Sign-in interrupted; nothing was saved.", "warn");
    cancelLogin(true);
  }

  function init() {
    $("saveKeyBtn").addEventListener("click", saveApiKey);
    $("clearKeyBtn").addEventListener("click", clearApiKey);
    $("connectBtn").addEventListener("click", startLogin);
    $("cancelLoginBtn").addEventListener("click", function () {
      cancelLogin(true);
    });
    $("logoutBtn").addEventListener("click", function () {
      request("/orcarouter/api/logout", { method: "POST" }).then(function () {
        setStatus($("apiKeyStatus"), "Credential forgotten.", "ok");
        state.selectedModel = null;
        $("modelTriggerLabel").textContent = "Select a model…";
        $("selectedModel").hidden = true;
        refreshStatus().then(loadModels);
      });
    });
    $("refreshModelsBtn").addEventListener("click", function () {
      request(
        "/orcarouter/api/models?capability=" +
          encodeURIComponent($("capabilitySelect").value) +
          "&modality=" +
          encodeURIComponent($("modalitySelect").value) +
          "&refresh=1"
      ).then(function () {
        loadModels();
      });
    });
    $("capabilitySelect").addEventListener("change", loadModels);
    $("modalitySelect").addEventListener("change", loadModels);

    $("modelTrigger").addEventListener("click", function () {
      if (state.dropdownOpen) {
        closeDropdown();
      } else {
        openDropdown();
      }
    });
    document.addEventListener("click", function (event) {
      if (!state.dropdownOpen) return;
      if ($("modelSelector").contains(event.target)) return;
      closeDropdown();
    });
    document.addEventListener("keydown", function (event) {
      if (event.key === "Escape" && state.dropdownOpen) {
        closeDropdown();
      }
    });

    // Every terminal path releases the login attempt.
    window.addEventListener("pagehide", onPageHide);
    window.addEventListener("beforeunload", onPageHide);

    setLoginBusy(false, "");
    refreshStatus().then(loadModels);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }

  // Exposed for the headless UI check.
  window.__orca = {
    state: state,
    loadModels: loadModels,
    openDropdown: openDropdown,
    closeDropdown: closeDropdown,
    onPageHide: onPageHide,
  };
})();
