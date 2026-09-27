const API = "/api";

let projects = [];
let appState = null;
let currentProjectId = null;
let activeBranchId = null;
let selectedNodeId = null;
let selectedChoiceId = null;
let composerMode = "write";
let currentDraft = null;
let currentGenerationLogId = null;
let currentGenerationMetrics = null;
let currentGenerationRequest = null;
let healthState = null;
let editingProject = false;
let expandedMapNodeId = null;
let currentGenerationContext = null;
let currentGenerationValidationPass = true;
let currentGenerationIssues = [];
let currentDraftValidation = null;
let draftValidationTimerId = null;
let storyElementCatalogData = [];
let draftStoryElementSelections = new Map();
let storyElementsReturnView = "home";
let editingStoryElementId = null;
let editingStoryElementSnapshot = null;
let manualSceneImageLogId = null;
let draftSceneImageLogId = null;

let activeMediaRecorder = null;
let activeMediaStream = null;
let activeRecordingTargetId = null;
let recordingChunks = [];
let recordingStartedAt = 0;
let recordingTimerId = null;
const transcriptionLogsByTarget = new Map();

const MAP_NODE_WIDTH = 210;
const MAP_NODE_HEIGHT = 112;
const MAP_HORIZONTAL_GAP = 85;
const MAP_VERTICAL_GAP = 48;


const STORY_ELEMENT_DETAIL_FIELDS = {
  creature: [
    { key: "behaviour", label: "Behaviour", kind: "text" },
    { key: "attacks", label: "Attacks", kind: "list" },
    { key: "defences", label: "Defences", kind: "list" },
    { key: "possible_rewards", label: "Possible rewards / drops", kind: "list" }
  ],
  hazard: [
    { key: "activation", label: "Activation", kind: "text" },
    { key: "detection", label: "How it can be detected", kind: "list" },
    { key: "disarm", label: "How it can be disarmed / avoided", kind: "list" },
    { key: "effect", label: "Effect / damage", kind: "text" }
  ],
  item: [
    { key: "function", label: "Function / effect", kind: "text" },
    { key: "obtain", label: "How it can be obtained", kind: "list" }
  ],
  location: [
    { key: "atmosphere", label: "Atmosphere", kind: "text" },
    { key: "features", label: "Notable features", kind: "list" },
    { key: "dangers", label: "Possible dangers", kind: "list" },
    { key: "discoveries", label: "Possible discoveries", kind: "list" }
  ],
  npc: [
    { key: "role", label: "Role", kind: "text" },
    { key: "personality", label: "Personality", kind: "text" },
    { key: "motivations", label: "Motivations", kind: "list" },
    { key: "notable_traits", label: "Notable traits", kind: "list" }
  ],
  encounter: [
    { key: "trigger", label: "Trigger", kind: "text" },
    { key: "challenge", label: "Challenge", kind: "text" },
    { key: "developments", label: "Possible developments", kind: "list" },
    { key: "outcomes", label: "Possible outcomes", kind: "list" }
  ]
};



const DELTA_ARRAY_FIELDS = [
  ["active_characters_add", "Characters added"],
  ["active_characters_remove", "Characters removed"],
  ["inventory_add", "Inventory added"],
  ["inventory_remove", "Inventory removed"],
  ["goals_add", "Goals added"],
  ["goals_complete", "Goals completed"],
  ["unresolved_clues_add", "Clues added"],
  ["unresolved_clues_resolve", "Clues resolved"],
  ["decisions_add", "Decisions recorded"]
];

const STORY_MODEL_CONFIG = {
  "qwen2.5:1.5b": {
    label:
      "Qwen2.5 1.5B — Default structured writer",
    priority: 1
  },

  "qwen3:1.7b": {
    label:
      "Qwen3 1.7B — Fallback baseline",
    priority: 2
  },

  "qwen3.5:2b": {
    label:
      "Qwen3.5 2B — Experimental",
    priority: 3
  },

  "qwen3.5:4b": {
    label:
      "Qwen3.5 4B — Slow benchmark",
    priority: 4
  },

  "qwen3:4b-instruct": {
    label:
      "Qwen3 4B Instruct — Slow benchmark",
    priority: 5
  },

  "gemma3:4b": {
    label:
      "Gemma 3 4B — Slow benchmark",
    priority: 6
  }
};

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

async function apiRequest(path, options = {}) {
  const response = await fetch(`${API}${path}`, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options
  });
  let payload = null;
  const text = await response.text();
  if (text) {
    try { payload = JSON.parse(text); } catch { payload = text; }
  }
  if (!response.ok) {
    const detail = payload?.detail || payload || `HTTP ${response.status}`;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return payload;
}


async function apiFormRequest(path, formData) {
  const response = await fetch(`${API}${path}`, {
    method: "POST",
    body: formData
  });
  let payload = null;
  const text = await response.text();
  if (text) {
    try { payload = JSON.parse(text); } catch { payload = text; }
  }
  if (!response.ok) {
    const detail = payload?.detail || payload || `HTTP ${response.status}`;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return payload;
}

function commaList(value) {
  return value.split(",").map((item) => item.trim()).filter(Boolean);
}

function lines(value) {
  return value.split(/\r?\n/).map((item) => item.trim()).filter(Boolean);
}

function relationshipsFromText(value) {
  const result = {};
  for (const line of lines(value)) {
    const separator = line.indexOf(":");
    if (separator > 0) {
      const name = line.slice(0, separator).trim();
      const status = line.slice(separator + 1).trim();
      if (name && status) result[name] = status;
    }
  }
  return result;
}

function relationshipsToText(value = {}) {
  return Object.entries(value).map(([name, status]) => `${name}: ${status}`).join("\n");
}

function formatValue(value) {
  if (Array.isArray(value)) return value.length ? value.join(", ") : "None";
  if (value && typeof value === "object") {
    const entries = Object.entries(value);
    return entries.length ? entries.map(([key, val]) => `${key}: ${val}`).join("\n") : "None";
  }
  return value || "None";
}

function formatDate(value) {
  if (!value) return "";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString([], { dateStyle: "medium", timeStyle: "short" });
}

function openModal(id) { document.getElementById(id).classList.remove("hidden"); }
function closeModal(id) { document.getElementById(id).classList.add("hidden"); }

function getProject() { return appState?.project || null; }
function getBranch(id = activeBranchId) { return appState?.branches.find((branch) => branch.id === id) || null; }
function getNode(id = selectedNodeId) { return appState?.nodes.find((node) => node.id === id) || null; }
function getBranchPath(branchId = activeBranchId) {
  const ids = appState?.branch_paths?.[branchId] || [];
  return ids.map((id) => getNode(id)).filter(Boolean);
}
function branchContainsNode(branchId, nodeId) {
  return (appState?.branch_paths?.[branchId] || []).includes(nodeId);
}
function getSelectedChoice() {
  const node = getNode();
  return node?.choices?.find((choice) => choice.id === selectedChoiceId) || null;
}

async function refreshHealth() {
  try {
    healthState = await apiRequest("/health");

    const installedModels = healthState.installed_models || [];

    const ollamaReady = healthState.ollama_available;

    const writerReady =
      installedModels.some(
        (model) =>
          Object.hasOwn(
            STORY_MODEL_CONFIG,
            model
          )
      );

    const embeddingReady = healthState.embedding_available;

    const continuityReady = healthState.continuity_available;

    const sttReady = Boolean(healthState.stt_dependency_available && healthState.stt_model_cached);

    const pipelineReady =
      ollamaReady &&
      writerReady &&
      embeddingReady &&
      continuityReady;

    const text = pipelineReady
      ? `AI pipeline ready · Writer + Embedding + NLI · STT ${sttReady ? "ready" : "setup needed"}`
      : [
          ollamaReady ? "Ollama ready" : "Ollama unavailable",
          writerReady ? "Writer ready": "Writer missing",
          embeddingReady ? "Embedding ready" : "Embedding missing",
          continuityReady ? "NLI ready" : "NLI missing",
          sttReady ? "STT ready" : "STT setup needed"
        ].join(" · ");

    for (const id of ["homeServiceStatus", "workspaceServiceStatus"]) {
      const badge = document.getElementById(id);
      badge.className = `status-pill ${pipelineReady ? "pass" : "warn"}`;
      badge.textContent = text;
    }

    populateModels();
  } catch (error) {
    for (const id of ["homeServiceStatus", "workspaceServiceStatus"]) {
      const badge = document.getElementById(id);
      badge.className = "status-pill fail";
      badge.textContent = `Service error: ${error.message}`;
    }
  }
}

function populateModels() {
  const select =
    document.getElementById("aiModel");

  const generateButton =
    document.getElementById(
      "generateDraftBtn"
    );

  const previousModel = select.value;

  /*
   * Set removes accidental duplicate names.
   */
  const installedModels = [
    ...new Set(
      healthState?.installed_models || []
    )
  ];

  /*
   * Only installed models that SekAI has approved
   * for the story-writer role are retained.
   */
  const writerModels = installedModels
    .filter(
      (model) =>
        Object.hasOwn(
          STORY_MODEL_CONFIG,
          model
        )
    )
    .sort(
      (left, right) =>
        STORY_MODEL_CONFIG[left].priority -
        STORY_MODEL_CONFIG[right].priority
    );

  select.replaceChildren();

  if (writerModels.length === 0) {
    const option =
      document.createElement("option");

    option.value = "";
    option.textContent =
      "No compatible story model installed";

    select.appendChild(option);
    select.disabled = true;
    generateButton.disabled = true;
    return;
  }

  for (const model of writerModels) {
    const option =
      document.createElement("option");

    option.value = model;
    option.textContent =
      STORY_MODEL_CONFIG[model].label;

    select.appendChild(option);
  }

  select.disabled = false;
  generateButton.disabled = false;

  if (writerModels.includes(previousModel)) {
    select.value = previousModel;
  } else {
    select.value = writerModels[0];
  }
}

async function loadProjects() {
  projects = await apiRequest("/projects");
  renderHomeStories();
  renderSidebarStories();
}

function renderHomeStories() {
  const grid = document.getElementById("storyGrid");
  const query = document.getElementById("storySearch").value.trim().toLowerCase();
  const filtered = projects.filter((project) =>
    [project.title, project.genre, project.tone, project.preview]
      .join(" ").toLowerCase().includes(query)
  );

  if (!filtered.length) {
    grid.innerHTML = `<div class="empty-home"><h3>${projects.length ? "No stories match your search" : "No stories yet"}</h3><p>Create a blank campaign story to begin writing.</p></div>`;
    return;
  }

  grid.innerHTML = filtered.map((project) => `
    <article class="story-card" data-project-id="${escapeHtml(project.id)}" tabindex="0">
      <div>
        <p class="eyebrow">${escapeHtml(project.genre)} · ${escapeHtml(project.tone)}</p>
        <h3>${escapeHtml(project.title)}</h3>
      </div>
      <p class="story-card-preview">${escapeHtml(project.preview || "Open this story to begin writing.")}</p>
      <div class="story-card-meta">
        <span>${project.node_count} node${project.node_count === 1 ? "" : "s"}</span>
        <span>${project.branch_count} branch${project.branch_count === 1 ? "" : "es"}</span>
        <span>Updated ${escapeHtml(formatDate(project.updated_at))}</span>
      </div>
      <div class="story-card-actions">
        <strong>Open story</strong>
        <button class="story-card-delete" data-delete-project="${escapeHtml(project.id)}" type="button">Delete</button>
      </div>
    </article>
  `).join("");

  grid.querySelectorAll("[data-project-id]").forEach((card) => {
    const open = () => openProject(card.dataset.projectId);
    card.addEventListener("click", (event) => {
      if (!event.target.closest("[data-delete-project]")) open();
    });
    card.addEventListener("keydown", (event) => {
      if (event.key === "Enter") open();
    });
  });

  grid.querySelectorAll("[data-delete-project]").forEach((button) => {
    button.addEventListener("click", async (event) => {
      event.stopPropagation();
      const project = projects.find((item) => item.id === button.dataset.deleteProject);
      if (!confirm(`Delete '${project?.title || "this story"}' and all of its branches?`)) return;
      try {
        await apiRequest(`/projects/${encodeURIComponent(button.dataset.deleteProject)}`, { method: "DELETE" });
        if (currentProjectId === button.dataset.deleteProject) showHome();
        await loadProjects();
      } catch (error) {
        alert(`Could not delete story: ${error.message}`);
      }
    });
  });
}

function renderSidebarStories() {
  const container = document.getElementById("sidebarStories");
  container.innerHTML = projects.map((project) => `
    <button type="button" class="sidebar-story-item ${project.id === currentProjectId ? "active" : ""}" data-sidebar-project="${escapeHtml(project.id)}">
      <strong>${escapeHtml(project.title)}</strong>
      <small>${project.node_count} nodes · ${project.branch_count} branches</small>
    </button>
  `).join("");
  container.querySelectorAll("[data-sidebar-project]").forEach((button) => {
    button.addEventListener("click", () => openProject(button.dataset.sidebarProject));
  });
}

function showHome() {
  currentProjectId = null;
  appState = null;
  activeBranchId = null;
  selectedNodeId = null;
  expandedMapNodeId = null;
  selectedChoiceId = null;
  document.getElementById("workspaceView").classList.add("hidden");
  document.getElementById("homeView").classList.remove("hidden");
  loadProjects().catch(console.error);
}

async function openProject(projectId) {
  try {
    appState = await apiRequest(`/projects/${encodeURIComponent(projectId)}`);
    currentProjectId = projectId;
    activeBranchId = appState.branches[0]?.id || null;
    selectedNodeId = getBranch(activeBranchId)?.head_node_id || appState.nodes[0]?.id || null;
    selectedChoiceId = null;
    clearDraft();
    expandedMapNodeId = null;
    document.getElementById("homeView").classList.add("hidden");
    document.getElementById("workspaceView").classList.remove("hidden");
    renderSidebarStories();
    renderWorkspace();
    await renderObjectives();
  } catch (error) {
    alert(`Could not open story: ${error.message}`);
  }
}

function renderWorkspace() {
  const project = getProject();
  if (!project) return;
  document.getElementById("workspaceTitle").textContent = project.title;
  document.getElementById("workspaceGenreTone").textContent = `${project.genre} · ${project.tone}`;
  document.getElementById("workspaceObjective").textContent = project.main_objective || "No main objective recorded.";
  renderBranchFilter();
  renderStoryMap();
  renderSelectedNode();
  renderMemory();
  renderComposerContext();
  renderCompareSelectors();

  const deleteBranchButton = document.getElementById("deleteBranchBtn");
  const branch = getBranch(activeBranchId);
  deleteBranchButton.disabled = !branch?.parent_branch_id;
  deleteBranchButton.title = branch?.parent_branch_id
    ? "Move this fork and its descendant branches to Trash"
    : "The main branch cannot be deleted";
}

function renderBranchFilter() {
  const container = document.getElementById("branchFilter");
  container.innerHTML = appState.branches.map((branch) => `
    <button type="button" class="branch-chip ${branch.id === activeBranchId ? "active" : ""}" data-branch-filter="${escapeHtml(branch.id)}">${escapeHtml(branch.name)}</button>
  `).join("");
  container.querySelectorAll("[data-branch-filter]").forEach((button) => {
    button.addEventListener("click", () => {
      activeBranchId = button.dataset.branchFilter;
      selectedNodeId = getBranch()?.head_node_id || selectedNodeId;
      selectedChoiceId = null;
      clearDraft();
      renderWorkspace();
    });
  });
}

function computeTreeLayout(nodes) {
  const children = {};
  const byId = Object.fromEntries(nodes.map((node) => [node.id, node]));
  for (const node of nodes) {
    if (node.parent_node_id) (children[node.parent_node_id] ||= []).push(node.id);
  }
  for (const childIds of Object.values(children)) {
    childIds.sort((a, b) => byId[a].created_at.localeCompare(byId[b].created_at));
  }
  const roots = nodes.filter((node) => !node.parent_node_id);
  let leafIndex = 0;
  let maxDepth = 0;
  const positions = {};

  function place(nodeId, depth) {
    maxDepth = Math.max(maxDepth, depth);
    const childIds = children[nodeId] || [];
    let y;
    if (!childIds.length) {
      y = 45 + leafIndex * (MAP_NODE_HEIGHT + MAP_VERTICAL_GAP);
      leafIndex += 1;
    } else {
      const childYs = childIds.map((childId) => place(childId, depth + 1));
      y = childYs.reduce((sum, value) => sum + value, 0) / childYs.length;
    }
    positions[nodeId] = {
      x: 35 + depth * (MAP_NODE_WIDTH + MAP_HORIZONTAL_GAP),
      y
    };
    return y;
  }

  roots.forEach((root) => place(root.id, 0));
  return {
    positions,
    children,
    width: Math.max(
      800,
      70 + (maxDepth + 1) * (MAP_NODE_WIDTH + MAP_HORIZONTAL_GAP)
    ),
    height: Math.max(
      330,
      90 + Math.max(leafIndex, 1) * (MAP_NODE_HEIGHT + MAP_VERTICAL_GAP)
    )
  };
}

function renderStoryMap() {
  const canvas = document.getElementById("storyMapCanvas");
  const nodes = appState.nodes;
  if (!nodes.length) {
    canvas.innerHTML = `<p class="subtle">No nodes yet.</p>`;
    return;
  }
  const layout = computeTreeLayout(nodes);
  canvas.style.width = `${layout.width}px`;
  canvas.style.height = `${layout.height}px`;
  const heads = new Set(appState.branches.map((branch) => branch.head_node_id));

  const edges = nodes.filter((node) => node.parent_node_id).map((node) => {
    const parent = layout.positions[node.parent_node_id];
    const child = layout.positions[node.id];
    const startX = parent.x + MAP_NODE_WIDTH;
    const startY = parent.y + MAP_NODE_HEIGHT / 2;
    const endX = child.x;
    const endY = child.y + MAP_NODE_HEIGHT / 2;
    const midX = (startX + endX) / 2;
    return `<path class="story-edge" d="M ${startX} ${startY} C ${midX} ${startY}, ${midX} ${endY}, ${endX} ${endY}" />`;
  }).join("");

  const nodeHtml = nodes.map((node) => {
    const position = layout.positions[node.id];
    const branch = getBranch(node.branch_id);
    const preview = String(node.story_text || "")
      .replace(/\s+/g, " ")
      .trim();
    const expanded = node.id === expandedMapNodeId;
    const imageThumb = node.scene_image?.image_url
      ? `
        <img class="map-node-thumb"
          src="${escapeHtml(node.scene_image.image_url)}"
          alt=""
        />
      `: "";
      
    return `
      <button
        type="button"
        class="map-node ${node.id === selectedNodeId ? "selected" : ""} ${heads.has(node.id) ? "branch-head" : ""} ${expanded ? "expanded" : ""}"
        data-map-node="${escapeHtml(node.id)}"
        style="left:${position.x}px;top:${position.y}px"
        title="${escapeHtml(node.title)}"
        aria-expanded="${expanded}"
      >
      ${imageThumb}
        <span class="map-mode-badge ${node.authoring_mode === "ai" ? "ai" : ""}">${escapeHtml(node.authoring_mode)} · ${escapeHtml(node.interaction_type)}</span>
        <strong class="map-node-title">${escapeHtml(node.title)}</strong>
        <span class="map-node-preview">${escapeHtml(preview)}</span>
        <small>${escapeHtml(branch?.name || "Shared path")}</small>
      </button>
    `;
  }).join("");

  canvas.innerHTML = `<svg class="story-map-svg" width="${layout.width}" height="${layout.height}" aria-hidden="true">${edges}</svg>${nodeHtml}`;
  canvas.querySelectorAll("[data-map-node]").forEach((button) => {
    button.addEventListener("click", () => {
      const nodeId = button.dataset.mapNode;
      expandedMapNodeId = expandedMapNodeId === nodeId ? null : nodeId;
      selectMapNode(nodeId);
    });
  });
}

function selectMapNode(nodeId) {
  selectedNodeId = nodeId;
  selectedChoiceId = null;
  if (!branchContainsNode(activeBranchId, nodeId)) {
    // A unique node records the branch that owns it. Prefer that branch over
    // the first branch that happens to contain the node in a shared path.
    const owningBranch = getBranch(getNode(nodeId)?.branch_id);
    const branchContaining = appState.branches.find(
      (branch) => branchContainsNode(branch.id, nodeId)
    );
    activeBranchId = owningBranch?.id || branchContaining?.id || activeBranchId;
  }
  clearDraft();
  renderWorkspace();
}

function renderSelectedNode() {
  const node = getNode();

  if (!node) {
    return;
  }

  const imagePreview = document.getElementById("selectedNodeImagePreview");
  const imageButton = document.getElementById("regenerateSelectedNodeImageBtn");
  const imageStatus = document.getElementById("selectedNodeImageStatus");

  if (node.scene_image?.image_url) {
    imagePreview.innerHTML = `
      <img src="${escapeHtml(node.scene_image.image_url)}"
        alt="Illustration for ${escapeHtml(node.title)}"
      />

      <small>
        ${escapeHtml(node.scene_image.model_key)}·
        ${escapeHtml(node.scene_image.device)}·
        ${Number(node.scene_image.inference_seconds).toFixed(2)} s
      </small>
    `;

    imageButton.textContent = "Regenerate image";

    imageStatus.textContent = "This node has a saved illustration.";
  } else {
    imagePreview.innerHTML = "";
    imageButton.textContent = "Generate image";
    imageStatus.textContent = "This node does not have an illustration.";
  }

  const branch = getBranch(activeBranchId);
  document.getElementById("selectedNodeTitle").textContent = node.title;
  document.getElementById("selectedNodeMeta").textContent = `${branch?.name || "Story"} · ${node.authoring_mode === "ai" ? `Generated by ${node.generated_by_model || "AI"}` : "Written by user"} · ${formatDate(node.updated_at)}`;
  const selectedNodeContent = document.getElementById("selectedNodeContent");

  selectedNodeContent.replaceChildren();

  if (node.scene_image?.image_url) {
    const image = document.createElement("img");

    image.className = "selected-node-image";
    image.src = node.scene_image.image_url;
    image.alt = `Illustration for ${node.title}`;
    selectedNodeContent.appendChild(image);
  }

  const story = document.createElement("div");
  story.className = "selected-node-story";
  story.textContent = node.story_text;
  selectedNodeContent.appendChild(story);

  const choicesContainer = document.getElementById("selectedNodeChoices");
  if (!node.choices?.length) {
    choicesContainer.innerHTML = `<p class="subtle">No prepared choices. Continue freely using the composer below.</p>`;
  } else {
    choicesContainer.innerHTML =
      node.choices.map(
        (choice, index) => `
          <article
            class="choice-card ${
              choice.id === selectedChoiceId
                ? "selected"
                : ""
            }"
          >
            <div class="choice-card-heading">
              <strong>
                ${index + 1}.
                ${escapeHtml(choice.label)}
              </strong>

              <span class="choice-type">
                ${escapeHtml(
                  choice.action_type || "do"
                )}
              </span>
            </div>

            <p class="subtle">
              Select this option, then optionally
              describe how you want the next scene
              to unfold.
            </p>

            <button
              type="button"
              class="ghost-button select-choice-button"
              data-choice-id="${escapeHtml(choice.id)}"
            >
              ${
                choice.id === selectedChoiceId
                  ? "Selected for next node"
                  : "Choose this path"
              }
            </button>
          </article>
        `
      ).join("");
    choicesContainer.querySelectorAll("[data-choice-id]").forEach((button) => {
      button.addEventListener("click", () => {
        selectedChoiceId =
          selectedChoiceId === button.dataset.choiceId
            ? null
            : button.dataset.choiceId;

        const selectedChoice = getSelectedChoice();

        if (selectedChoice) {
          setComposerMode(
            selectedChoice.action_type ||
            "continue"
          );
        } else {
          clearDraft();
        }

        renderSelectedNode();
        renderComposerContext();
      });
    });
  }
}

function renderMemory() {
  const node = getNode();

  const textarea = document.getElementById("memoryJsonView");

  if (!node) {
    textarea.value = "{}";
    return;
  }

  textarea.value =
    JSON.stringify(node.memory_snapshot || {}, null, 2);
}

function renderComposerContext() {
  const badge = document.getElementById("composerContextBadge");
  const node = getNode();
  const branch = getBranch();
  if (!node || !branch) {
    badge.textContent = "Select a node";
    return;
  }
  const forceFork = composerMode === "write"
    ? document.getElementById("manualForceFork")?.checked
    : document.getElementById("aiForceFork")?.checked;
  const fork = branch.head_node_id !== node.id || forceFork;
  const choice = getSelectedChoice();
  badge.className = `status-pill ${fork ? "warn" : "neutral"}`;
  badge.textContent = `${fork ? "Forking" : "Continuing"} from ${node.title}${choice ? ` · ${choice.label}` : ""}`;
}

function setComposerMode(mode) {
  composerMode = mode;

  document
    .querySelectorAll(".composer-tab")
    .forEach((button) =>
      button.classList.toggle(
        "active",
        button.dataset.composerMode === mode
      )
    );

  const isWrite = mode === "write";

  document
    .getElementById("manualComposer")
    .classList.toggle(
      "hidden",
      !isWrite
    );

  document
    .getElementById("aiComposer")
    .classList.toggle(
      "hidden",
      isWrite
    );

  const selectedChoice =
    getSelectedChoice();

  const label =
    document.getElementById(
      "playerInputLabel"
    );

  const input =
    document.getElementById(
      "playerInput"
    );

  if (selectedChoice && mode !== "write") {
    label.textContent = "Next-scene refinement";

    input.placeholder =
      `Selected choice: "${selectedChoice.label}". ` +
      "Optionally add details for how this choice should unfold. " +
      "If provided, these details will be treated as required.";
  }
  else {
    const labels = {
      continue:
        "Optional continuation prompt",

      do:
        "What does the player DO?",

      say:
        "What does the player SAY?",

      ask:
        "What does the player ASK?"
    };

    label.textContent =
      labels[mode] ||
      "Instruction";

    input.placeholder =
      mode === "continue"
        ? "Leave blank to continue naturally, or add a short direction."
        : mode === "say"
          ? "Describe what the player says."
          : mode === "ask"
            ? "Describe what the player asks."
            : "Describe what the player attempts.";
  }

  clearDraft();
  renderComposerContext();
}

function emptyDelta() {
  return {
    location_set: null,
    active_characters_add: [], active_characters_remove: [],
    inventory_add: [], inventory_remove: [],
    relationships_set: {}, goals_add: [], goals_complete: [],
    unresolved_clues_add: [], unresolved_clues_resolve: [],
    decisions_add: [], threat_set: null
  };
}

function deltaEditorHtml(prefix, delta = {}) {
  const current = { ...emptyDelta(), ...delta };
  return `
    <div><label for="${prefix}_location_set">Location set</label><input id="${prefix}_location_set" value="${escapeHtml(current.location_set || "")}" placeholder="No change" /></div>
    <div><label for="${prefix}_threat_set">Threat set</label><select id="${prefix}_threat_set"><option value="">No change</option>${["Low", "Medium", "High"].map((level) => `<option value="${level}" ${current.threat_set === level ? "selected" : ""}>${level}</option>`).join("")}</select></div>
    ${DELTA_ARRAY_FIELDS.map(([field, label]) => `<div><label for="${prefix}_${field}">${escapeHtml(label)}, comma-separated</label><input id="${prefix}_${field}" value="${escapeHtml((current[field] || []).join(", "))}" /></div>`).join("")}
    <div class="span-two"><label for="${prefix}_relationships_set">Relationships, one per line as Name: status</label><textarea id="${prefix}_relationships_set">${escapeHtml(relationshipsToText(current.relationships_set || {}))}</textarea></div>
  `;
}

function readDelta(prefix) {
  const delta = {
    location_set: document.getElementById(`${prefix}_location_set`)?.value.trim() || null,
    relationships_set: relationshipsFromText(document.getElementById(`${prefix}_relationships_set`)?.value || ""),
    threat_set: document.getElementById(`${prefix}_threat_set`)?.value || null
  };
  for (const [field] of DELTA_ARRAY_FIELDS) {
    delta[field] = commaList(document.getElementById(`${prefix}_${field}`)?.value || "");
  }
  return delta;
}

function renderManualChoicesEditor() {
  const container =
    document.getElementById(
      "manualChoicesEditor"
    );

  container.innerHTML =
    [0, 1, 2]
      .map(
        (index) => `
          <article class="choice-editor">

            <h4>
              Optional choice ${index + 1}
            </h4>

            <label
              for="manual_choice_${index}_type"
            >
              Type
            </label>

            <select
              id="manual_choice_${index}_type"
            >
              <option value="do">DO</option>
              <option value="ask">ASK</option>
              <option value="say">SAY</option>
              <option value="continue">
                CONTINUE
              </option>
            </select>

            <label
              for="manual_choice_${index}_label"
            >
              Choice text
            </label>

            <input
              id="manual_choice_${index}_label"
              type="text"
            />

          </article>
        `
      )
      .join("");
}

function collectManualChoices() {
  const choices = [];

  for (
    let index = 0;
    index < 3;
    index += 1
  ) {
    const label =
      document
        .getElementById(
          `manual_choice_${index}_label`
        )
        .value
        .trim();

    if (!label) {
      continue;
    }

    choices.push({
      action_type:
        document
          .getElementById(
            `manual_choice_${index}_type`
          )
          .value,

      label
    });
  }

  return choices;
}

function manualDelta() {
  const raw =
    document.getElementById("manualDeltaJson").value.trim();

  if (!raw) {
    return emptyDelta();
  }

  let parsed;

  try {
    parsed = JSON.parse(raw);
  } catch (error) {
    throw new Error(
      "Memory changes must contain ",
      "valid JSON."
    );
  }

  return {
    ...emptyDelta(),
    ...parsed
  };
}

async function saveManualNode() {
  const title = document.getElementById("manualTitle").value.trim();
  const storyText = document.getElementById("manualSceneText").value.trim();
  if (!selectedNodeId || !activeBranchId) return alert("Select a story node first.");
  if (!title || !storyText) return alert("Enter a node title and scene text.");
  const button = document.getElementById("saveManualNodeBtn");
  button.disabled = true;
  try {
    // Persist the exact human-edited manual scene transcript before saving.
    await recordReviewedTranscriptions(["manualSceneText"]);

    const result = await apiRequest("/nodes", {
      method: "POST",
      body: JSON.stringify({
        project_id: currentProjectId,
        source_branch_id: activeBranchId,
        parent_node_id: selectedNodeId,
        selected_choice_id: selectedChoiceId,
        branch_name: document.getElementById("manualBranchName").value.trim() || null,
        force_fork: document.getElementById("manualForceFork").checked,
        title,
        story_text: storyText,
        applied_delta: manualDelta(),
        choices: collectManualChoices(),
        authoring_mode: "manual",
        interaction_type: "write",
        interaction_text: storyText,
        generated_by_model: null,
        generation_log_id: null,
        scene_image_log_id: manualSceneImageLogId
      })
    });
    appState = result.state;
    activeBranchId = result.branch.id;
    selectedNodeId = result.node.id;
    selectedChoiceId = null;
    resetManualComposer();
    await loadProjects();
    renderWorkspace();
    await renderObjectives();
    alert(result.fork_created ? "Written scene saved as a new branch." : "Written scene saved as the next node.");
  } catch (error) {
    alert(`Could not save written scene: ${error.message}`);
  } finally {
    button.disabled = false;
  }
}

function resetManualComposer() {
  const title = document.getElementById("manualTitle");
  const sceneText = document.getElementById("manualSceneText");
  const branchName = document.getElementById("manualBranchName");
  const forceFork = document.getElementById("manualForceFork");
  const deltaEditor = document.getElementById("manualDeltaJson");
  const imagePrompt = document.getElementById("manualImagePrompt");

  if (title) {title.value = "";}
  if (sceneText) {sceneText.value = "";}
  if (branchName) {branchName.value = "";}
  if (forceFork) {forceFork.checked = false;}

  if (deltaEditor) {
    deltaEditor.value =
      JSON.stringify(
        emptyDelta(),
        null,
        2
      );
  }

  if (imagePrompt) {imagePrompt.value = "";}
  transcriptionLogsByTarget.delete("manualSceneText");
  setDictationStatus("manualSceneText",dictationDefaultStatus("manualSceneText"));
  renderManualChoicesEditor();
  clearGeneratedSceneImage(
    "manual"
  );
}


const DICTATION_TARGETS = {
  manualSceneText: {
    field: "manual_scene",
    statusId: "manualSceneDictationStatus",
    buttonLabel: "Dictate scene text",
    reviewAction: "saving"
  },
  playerInput: {
    field: "player_input",
    statusId: "playerInputDictationStatus",
    buttonLabel: "Dictate player input",
    reviewAction: "generation"
  },
  gmInstruction: {
    field: "gm_instruction",
    statusId: "gmInstructionDictationStatus",
    buttonLabel: "Dictate GM instruction",
    reviewAction: "generation"
  }
};

function dictationConfig(targetId) {
  const config = DICTATION_TARGETS[targetId];
  if (!config) {
    throw new Error(`Unsupported dictation target: ${targetId}`);
  }
  return config;
}

function dictationTargetField(targetId) {
  return dictationConfig(targetId).field;
}

function dictationStatusElement(targetId) {
  return document.getElementById(dictationConfig(targetId).statusId);
}

function dictationDefaultStatus(targetId) {
  const action = dictationConfig(targetId).reviewAction;
  return `Local speech-to-text · review transcript before ${action}`;
}

function dictationButton(targetId) {
  return document.querySelector(`[data-dictate-target="${targetId}"]`);
}

function setDictationStatus(targetId, text, className = "") {
  const status = dictationStatusElement(targetId);
  status.className = `dictation-status ${className}`.trim();
  status.textContent = text;
}

function chooseRecordingMimeType() {
  const candidates = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/ogg;codecs=opus",
    "audio/mp4"
  ];
  return candidates.find((type) => MediaRecorder.isTypeSupported(type)) || "";
}

function recordingFileName(mimeType) {
  if (mimeType.includes("ogg")) return "sekai-voice.ogg";
  if (mimeType.includes("mp4")) return "sekai-voice.m4a";
  return "sekai-voice.webm";
}

function stopRecordingTimer() {
  if (recordingTimerId !== null) {
    window.clearInterval(recordingTimerId);
    recordingTimerId = null;
  }
}

function releaseMicrophone() {
  stopRecordingTimer();
  if (activeMediaStream) {
    activeMediaStream.getTracks().forEach((track) => track.stop());
  }
  activeMediaStream = null;
}

async function submitRecording(targetId, blob) {
  const target = document.getElementById(targetId);
  const button = dictationButton(targetId);
  setDictationStatus(targetId, "Transcribing locally…");
  button.disabled = true;

  try {
    const formData = new FormData();
    formData.append("file", blob, recordingFileName(blob.type));
    formData.append("target_field", dictationTargetField(targetId));
    if (currentProjectId) formData.append("project_id", currentProjectId);

    const result = await apiFormRequest("/transcribe", formData);
    target.value = result.text;
    transcriptionLogsByTarget.set(targetId, result.transcription_log_id);

    const rtfText = result.real_time_factor === null || result.real_time_factor === undefined
      ? "RTF unavailable"
      : `RTF ${Number(result.real_time_factor).toFixed(3)}`;
    const coldText = result.cold_start ? " · cold start" : "";
    const reviewAction = dictationConfig(targetId).reviewAction;
    setDictationStatus(
      targetId,
      `${result.engine} ${result.model} · ${result.inference_seconds}s inference · ${rtfText}${coldText} · review/edit before ${reviewAction}`,
      "pass"
    );
    clearDraft();
  } catch (error) {
    setDictationStatus(targetId, `Transcription failed: ${error.message}`, "fail");
  } finally {
    button.disabled = false;
  }
}

async function startDictation(targetId) {
  if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") {
    return alert("This browser does not provide the MediaRecorder microphone API.");
  }
  if (activeMediaRecorder && activeMediaRecorder.state === "recording") {
    return alert("Stop the current recording before starting another one.");
  }

  const target = document.getElementById(targetId);
  if (
    target.value.trim() &&
    !confirm("Voice transcription will replace the current text in this field. Continue?")
  ) {
    return;
  }

  try {
    activeMediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    activeRecordingTargetId = targetId;
    recordingChunks = [];
    recordingStartedAt = performance.now();

    const mimeType = chooseRecordingMimeType();
    activeMediaRecorder = mimeType
      ? new MediaRecorder(activeMediaStream, { mimeType })
      : new MediaRecorder(activeMediaStream);

    activeMediaRecorder.addEventListener("dataavailable", (event) => {
      if (event.data.size > 0) recordingChunks.push(event.data);
    });

    activeMediaRecorder.addEventListener("stop", async () => {
      const completedTarget = activeRecordingTargetId;
      const recordedType = activeMediaRecorder?.mimeType || mimeType || "audio/webm";
      const blob = new Blob(recordingChunks, { type: recordedType });
      releaseMicrophone();
      activeMediaRecorder = null;
      activeRecordingTargetId = null;
      recordingChunks = [];
      if (completedTarget && blob.size > 0) {
        await submitRecording(completedTarget, blob);
      } else if (completedTarget) {
        setDictationStatus(completedTarget, "No audio was captured.", "fail");
      }
    }, { once: true });

    activeMediaRecorder.start();
    const button = dictationButton(targetId);
    button.classList.add("recording");
    button.textContent = "■ Stop recording";
    setDictationStatus(targetId, "Recording… 0s", "recording");

    recordingTimerId = window.setInterval(() => {
      if (!activeMediaRecorder || activeMediaRecorder.state !== "recording") return;
      const seconds = Math.floor((performance.now() - recordingStartedAt) / 1000);
      setDictationStatus(targetId, `Recording… ${seconds}s / 60s`, "recording");
      if (seconds >= 60) stopDictation();
    }, 250);
  } catch (error) {
    releaseMicrophone();
    activeMediaRecorder = null;
    activeRecordingTargetId = null;
    setDictationStatus(targetId, `Microphone unavailable: ${error.message}`, "fail");
  }
}

function stopDictation() {
  if (!activeMediaRecorder || activeMediaRecorder.state !== "recording") return;
  const targetId = activeRecordingTargetId;
  const button = targetId ? dictationButton(targetId) : null;
  if (button) {
    button.classList.remove("recording");
    button.textContent = dictationConfig(targetId).buttonLabel;
  }
  stopRecordingTimer();
  activeMediaRecorder.stop();
}

function toggleDictation(targetId) {
  if (
    activeMediaRecorder &&
    activeMediaRecorder.state === "recording" &&
    activeRecordingTargetId === targetId
  ) {
    stopDictation();
    return;
  }
  startDictation(targetId);
}

async function recordReviewedTranscriptions(
  targetIds = ["playerInput", "gmInstruction"]
) {
  const updates = [];
  for (const targetId of targetIds) {
    const logId = transcriptionLogsByTarget.get(targetId);
    if (!logId) continue;
    updates.push(
      apiRequest(`/transcriptions/${encodeURIComponent(logId)}`, {
        method: "PATCH",
        body: JSON.stringify({
          reviewed_text: document.getElementById(targetId).value.trim()
        })
      })
    );
  }
  if (!updates.length) return;
  const outcomes = await Promise.allSettled(updates);
  outcomes.forEach((outcome) => {
    if (outcome.status === "rejected") {
      console.warn("Could not persist reviewed transcription:", outcome.reason);
    }
  });
}

async function generateDraft() {
  const selectedChoice = getSelectedChoice();

  const interactionType = selectedChoice?.action_type || composerMode;
    
  if (!selectedNodeId || !activeBranchId) {
    return alert("Select a story node first.");
  }

  const playerInput = document
    .getElementById("playerInput")
    .value
    .trim();

  if (
    !selectedChoice &&
    ["do", "say", "ask"].includes(
      interactionType
    ) &&
    !playerInput
  ) {
    return alert(
      `Enter what the player will ${composerMode.toUpperCase()}.`
    );
  }

  // Persist the exact human-edited transcript before it enters Qwen.
  await recordReviewedTranscriptions();

  // Remove any earlier draft before starting a new request so a failed
  // generation cannot leave stale content that appears saveable.
  clearDraft();

  const requestBody = {
    project_id: currentProjectId,
    branch_id: activeBranchId,
    parent_node_id: selectedNodeId,
    selected_choice_id: selectedChoiceId,
    model: document.getElementById("aiModel").value,
    temperature: Number(
      document.getElementById("aiTemperature").value
    ),
    interaction_type: interactionType,
    player_input: playerInput,
    gm_instruction: document
      .getElementById("gmInstruction")
      .value
      .trim()
  };

  const button =
    document.getElementById("generateDraftBtn");

  button.disabled = true;

  document.getElementById(
    "generationStatus"
  ).textContent =
    "Generating with local Ollama model…";

  try {
    const payload = await apiRequest(
      "/generate-story",
      {
        method: "POST",
        body: JSON.stringify(requestBody)
      }
    );

    currentDraft = structuredClone(payload.draft);

    currentGenerationLogId = payload.generation_log_id;

    currentGenerationMetrics = structuredClone(payload.metrics);

    currentGenerationContext = structuredClone(payload.authoritative_context);
    currentGenerationValidationPass = Boolean(payload.validation_pass);
    currentGenerationIssues = structuredClone(payload.validation_issues || []);

    document.getElementById("aiContextJson").textContent =
        JSON.stringify(currentGenerationContext, null, 2);

    /*
     * Freeze the exact node, branch, interaction and
     * instruction that produced this draft.
     */
    currentGenerationRequest = structuredClone(requestBody);

    renderAiDraft(
      payload.draft,
      payload.metrics,
      payload.validation_issues || []
    );

  } catch (error) {
    currentGenerationRequest = null;

    document.getElementById(
      "generationStatus"
    ).textContent =
      `Generation failed: ${error.message}`;

  } finally {
    button.disabled = false;
  }
}

function renderAiDraft(
  draft,
  metrics,
  validationIssues = []
) {
  document
    .getElementById("aiDraftWorkspace")
    .classList.remove("hidden");

  document.getElementById("draftTitle").value = draft.title;
  document.getElementById("draftText").value = draft.story_text;

  const modelText = metrics.fallback_used
    ? `requested ${metrics.requested_model} → used ${metrics.model}`
    : metrics.model;

  const semanticText = metrics.semantic_check_completed
    ? "semantic repetition check passed"
    : "semantic repetition check unavailable";

  const continuityText = metrics.continuity_check_completed
    ? `NLI continuity checked ${metrics.continuity_checked_facts} facts${
        metrics.continuity_max_contradiction !== null &&
        metrics.continuity_max_contradiction !== undefined
          ? ` · max contradiction ${Number(
              metrics.continuity_max_contradiction
            ).toFixed(2)}`
          : ""
      }`
    : "NLI continuity check unavailable";

  document.getElementById("draftMetrics").textContent =
    `${metrics.elapsed_seconds}s` +
    `${metrics.tokens_per_second ? ` · ${metrics.tokens_per_second} tokens/s` : ""}` +
    ` · ${modelText}` +
    ` · ${semanticText}` +
    ` · ${continuityText}` +
    ` · temperature ${metrics.effective_temperature}` +
    `${metrics.attempts > 1 ? ` · ${metrics.attempts} attempts` : ""}`;

  document.getElementById("draftDeltaEditor").innerHTML = `
    <label for="draftDeltaJson">
      Memory changes proposed by AI
    </label>

    <textarea
      id="draftDeltaJson"
      class="large-textarea"
      spellcheck="false"
    >${escapeHtml(JSON.stringify(draft.applied_delta, null, 2))}</textarea>
  `;

  document.getElementById("draftChoicesEditor").innerHTML = draft.choices
    .map(
      (choice, index) => `
        <article class="choice-editor">
          <h4>Generated choice ${index + 1}</h4>

          <label for="draft_choice_${index}_type">Type</label>
          <select id="draft_choice_${index}_type">
            ${["do", "ask", "say", "continue"]
              .map(
                (type) => `
                  <option
                    value="${type}"
                    ${choice.action_type === type ? "selected" : ""}
                  >
                    ${type.toUpperCase()}
                  </option>
                `
              )
              .join("")}
          </select>

          <label for="draft_choice_${index}_label">Choice text</label>
          <input
            id="draft_choice_${index}_label"
            value="${escapeHtml(choice.label)}"
          />
        </article>
      `
    )
    .join("");

  const initialIssues = structuredClone(validationIssues || []);
  currentDraftValidation = {
    can_save: !initialIssues.some((issue) => issue.severity === "blocking"),
    has_warnings: initialIssues.some((issue) => issue.severity === "warning"),
    word_count: countWords(draft.story_text),
    blocking_count: initialIssues.filter((issue) => issue.severity === "blocking").length,
    warning_count: initialIssues.filter((issue) => issue.severity === "warning").length,
    issues: initialIssues
  };

  updateDraftWordCount();
  renderDraftValidation(currentDraftValidation);
  bindDraftValidationInputs();
}

function countWords(value) {
  return value.trim().match(/\b[\w'-]+\b/g)?.length || 0;
}

function updateDraftWordCount() {
  const text = document.getElementById("draftText")?.value || "";
  const count = countWords(text);
  const counter = document.getElementById("draftWordCount");

  if (!counter) {
    return;
  }

  counter.textContent = `${count} words`;
  counter.classList.toggle("warn", count < 50 || count > 150);
}

function clearDraftValidationStyles() {
  for (const id of ["draftTitle", "draftText", "draftDeltaJson"]) {
    const element = document.getElementById(id);
    element?.classList.remove("validation-warning", "validation-error");
  }
}

function validationTargetLabel(issue) {
  if (issue.target === "story_text") {
    return "Scene text";
  }
  if (issue.target === "title") {
    return "Node title";
  }
  if (issue.target === "choices") {
    return "Choices";
  }
  if (issue.target === "applied_delta") {
    return issue.field
      ? `Memory · ${issue.field.replaceAll("_", " ")}`
      : "Memory changes";
  }
  return issue.stage.replaceAll("_", " ");
}

function renderDraftValidation(validation) {
  currentDraftValidation = validation;
  clearDraftValidationStyles();

  const warningBox = document.getElementById("draftValidationWarning");
  const heading = document.getElementById("draftValidationHeading");
  const issueList = document.getElementById("draftValidationIssues");
  const saveButton = document.getElementById("saveAiDraftBtn");
  const textFeedback = document.getElementById("draftTextFeedback");
  const memoryFeedback = document.getElementById("draftMemoryValidation");
  const issues = validation?.issues || [];
  const blockingCount = issues.filter(
    (issue) => issue.severity === "blocking"
  ).length;
  const warningCount = issues.filter(
    (issue) => issue.severity === "warning"
  ).length;

  for (const issue of issues) {
    const className =
      issue.severity === "blocking"
        ? "validation-error"
        : "validation-warning";

    if (issue.target === "story_text") {
      document.getElementById("draftText")?.classList.add(className);
    } else if (issue.target === "title") {
      document.getElementById("draftTitle")?.classList.add(className);
    } else if (issue.target === "applied_delta") {
      document.getElementById("draftDeltaJson")?.classList.add(className);
    }
  }

  const storyIssues = issues.filter((issue) => issue.target === "story_text");
  if (textFeedback) {
    textFeedback.textContent = storyIssues.length
      ? [storyIssues[0].message, storyIssues[0].suggestion]
          .filter(Boolean)
          .join(" ")
      : "";
  }

  const memoryIssues = issues.filter((issue) => issue.target === "applied_delta");
  if (memoryFeedback) {
    memoryFeedback.innerHTML = memoryIssues
      .map(
        (issue) => `
          <div class="draft-memory-issue ${escapeHtml(issue.severity)}">
            <strong>${escapeHtml(validationTargetLabel(issue))}</strong>
            <span>${escapeHtml(issue.message)}</span>
          </div>
        `
      )
      .join("");
  }

  if (!issues.length) {
    warningBox.classList.add("hidden");
    issueList.innerHTML = "";
    saveButton.disabled = false;
    saveButton.textContent = "Save reviewed AI scene";
    document.getElementById("generationStatus").textContent =
      "Draft ready. Review and edit before saving.";
    return;
  }

  warningBox.classList.remove("hidden");
  warningBox.classList.toggle("blocking", blockingCount > 0);

  if (heading) {
    heading.textContent = blockingCount > 0
      ? `${blockingCount} blocking issue${blockingCount === 1 ? "" : "s"} must be fixed`
      : `${warningCount} warning${warningCount === 1 ? "" : "s"} to review`;
  }

  issueList.innerHTML = issues
    .map(
      (issue, index) => `
        <article class="draft-review-issue ${escapeHtml(issue.severity)}">
          <div class="draft-review-issue-heading">
            <span class="draft-review-field">
              ${escapeHtml(validationTargetLabel(issue))}
            </span>
            <span class="status-pill ${issue.severity === "blocking" ? "fail" : "warn"}">
              ${issue.severity === "blocking" ? "Must fix" : "Review"}
            </span>
          </div>
          <p>${escapeHtml(issue.message)}</p>
          ${
            issue.suggestion
              ? `<p class="draft-review-suggestion">${escapeHtml(issue.suggestion)}</p>`
              : ""
          }
          <div class="draft-review-actions">
            <button
              type="button"
              class="ghost-button"
              data-focus-draft-issue="${index}"
            >
              Show field
            </button>
            ${
              issue.target === "applied_delta" && issue.field && issue.value
                ? `
                  <button
                    type="button"
                    class="ghost-button"
                    data-remove-draft-issue="${index}"
                  >
                    Remove memory change
                  </button>
                `
                : ""
            }
          </div>
        </article>
      `
    )
    .join("");

  issueList.querySelectorAll("[data-focus-draft-issue]").forEach((button) => {
    button.addEventListener("click", () => {
      const issue = issues[Number(button.dataset.focusDraftIssue)];
      focusDraftIssue(issue);
    });
  });

  issueList.querySelectorAll("[data-remove-draft-issue]").forEach((button) => {
    button.addEventListener("click", () => {
      const issue = issues[Number(button.dataset.removeDraftIssue)];
      removeDraftDeltaIssue(issue);
    });
  });

  saveButton.disabled = blockingCount > 0;
  saveButton.textContent = blockingCount > 0
    ? `Fix ${blockingCount} blocking issue${blockingCount === 1 ? "" : "s"} to save`
    : "Save after human review";

  document.getElementById("generationStatus").textContent =
    blockingCount > 0
      ? "This draft cannot be saved yet. Fix the highlighted blocking issues, then SekAI will re-check it automatically."
      : "The draft has non-blocking warnings. Review the highlighted fields before saving or regenerate it.";
}

function focusDraftIssue(issue) {
  let target = null;

  if (issue.target === "story_text") {
    target = document.getElementById("draftText");
  } else if (issue.target === "title") {
    target = document.getElementById("draftTitle");
  } else if (issue.target === "applied_delta") {
    target = document.getElementById("draftDeltaJson");
    const details = target?.closest("details");
    if (details) {
      details.open = true;
    }
  } else if (issue.target === "choices") {
    target = document.getElementById("draftChoicesEditor");
  }

  target?.scrollIntoView({ behavior: "smooth", block: "center" });
  target?.focus?.();
}

function removeDraftDeltaIssue(issue) {
  if (issue.target !== "applied_delta" || !issue.field) {
    return;
  }

  const editor = document.getElementById("draftDeltaJson");
  let delta;

  try {
    delta = readDeltaJson("draftDeltaJson");
  } catch (error) {
    alert(error.message);
    return;
  }

  const field = issue.field;
  const value = issue.value;

  if (Array.isArray(delta[field])) {
    const expected = String(value || "").trim().toLowerCase();
    delta[field] = delta[field].filter(
      (item) => String(item).trim().toLowerCase() !== expected
    );
  } else if (field === "relationships_set" && value) {
    const matchingKey = Object.keys(delta.relationships_set || {}).find(
      (key) => key.trim().toLowerCase() === String(value).trim().toLowerCase()
    );
    if (matchingKey) {
      delete delta.relationships_set[matchingKey];
    }
  } else if (field === "location_set" || field === "threat_set") {
    delta[field] = null;
  }

  editor.value = JSON.stringify(delta, null, 2);
  scheduleDraftValidation(true);
}

function localDraftValidation(issues) {
  return {
    can_save: !issues.some((issue) => issue.severity === "blocking"),
    has_warnings: issues.some((issue) => issue.severity === "warning"),
    word_count: countWords(document.getElementById("draftText")?.value || ""),
    blocking_count: issues.filter((issue) => issue.severity === "blocking").length,
    warning_count: issues.filter((issue) => issue.severity === "warning").length,
    issues
  };
}

async function validateCurrentAiDraft() {
  if (!currentDraft || !currentGenerationRequest) {
    return null;
  }

  updateDraftWordCount();

  const title = document.getElementById("draftTitle").value.trim();
  const storyText = document.getElementById("draftText").value.trim();
  const choices = collectAiChoices();
  const localIssues = [];

  if (!title) {
    localIssues.push({
      stage: "title",
      severity: "blocking",
      target: "title",
      field: "title",
      value: null,
      message: "The node title cannot be empty.",
      suggestion: "Enter a short descriptive node title.",
      can_override: false
    });
  }

  if (!storyText) {
    localIssues.push({
      stage: "length",
      severity: "blocking",
      target: "story_text",
      field: "story_text",
      value: null,
      message: "The scene text cannot be empty.",
      suggestion: "Write or regenerate the scene before saving.",
      can_override: false
    });
  }

  if (choices.length !== 3 || choices.some((choice) => !choice.label)) {
    localIssues.push({
      stage: "choices",
      severity: "blocking",
      target: "choices",
      field: "choices",
      value: null,
      message: "A reviewed AI scene must contain three non-empty choices.",
      suggestion: "Complete all three choice labels before saving.",
      can_override: false
    });
  } else {
    const labels = choices.map((choice) => choice.label.trim().toLowerCase());
    if (new Set(labels).size !== 3) {
      localIssues.push({
        stage: "choices",
        severity: "blocking",
        target: "choices",
        field: "choices",
        value: null,
        message: "The three choice labels must be distinct.",
        suggestion: "Edit repeated choices so each leads in a different direction.",
        can_override: false
      });
    }
  }

  let appliedDelta;
  try {
    appliedDelta = readDeltaJson("draftDeltaJson");
  } catch (error) {
    localIssues.push({
      stage: "state",
      severity: "blocking",
      target: "applied_delta",
      field: "applied_delta",
      value: null,
      message: error.message,
      suggestion: "Correct the JSON syntax before saving.",
      can_override: false
    });
  }

  if (localIssues.length) {
    const validation = localDraftValidation(localIssues);
    renderDraftValidation(validation);
    return validation;
  }

  try {
    const validation = await apiRequest("/validate-ai-draft", {
      method: "POST",
      body: JSON.stringify({
        project_id: currentGenerationRequest.project_id,
        branch_id: currentGenerationRequest.branch_id,
        parent_node_id: currentGenerationRequest.parent_node_id,
        title,
        story_text: storyText,
        applied_delta: appliedDelta,
        choices
      })
    });

    const specialistWarnings = (currentGenerationIssues || []).filter(
      (issue) =>
        ["repetition", "semantic_repetition", "continuity"].includes(issue.stage)
    );

    if (specialistWarnings.length) {
      const existingKeys = new Set(
        validation.issues.map(
          (issue) => `${issue.stage}:${issue.message}`
        )
      );

      for (const issue of specialistWarnings) {
        const key = `${issue.stage}:${issue.message}`;
        if (!existingKeys.has(key)) {
          validation.issues.push({
            ...issue,
            severity: "warning",
            target: issue.target || "story_text",
            can_override: true,
            suggestion:
              issue.suggestion ||
              "This specialist check was produced during generation and is not rerun by the quick edit validator. Review the edited scene manually before saving."
          });
        }
      }

      validation.warning_count = validation.issues.filter(
        (issue) => issue.severity === "warning"
      ).length;
      validation.blocking_count = validation.issues.filter(
        (issue) => issue.severity === "blocking"
      ).length;
      validation.has_warnings = validation.warning_count > 0;
      validation.can_save = validation.blocking_count === 0;
    }

    renderDraftValidation(validation);
    return validation;
  } catch (error) {
    const validation = localDraftValidation([
      {
        stage: "state",
        severity: "blocking",
        target: "applied_delta",
        field: null,
        value: null,
        message: `Draft validation could not run: ${error.message}`,
        suggestion: "Retry validation before saving so the backend and editor agree.",
        can_override: false
      }
    ]);
    renderDraftValidation(validation);
    return validation;
  }
}

function scheduleDraftValidation(immediate = false) {
  if (draftValidationTimerId !== null) {
    clearTimeout(draftValidationTimerId);
  }

  const delay = immediate ? 0 : 450;
  draftValidationTimerId = window.setTimeout(() => {
    draftValidationTimerId = null;
    validateCurrentAiDraft().catch(console.error);
  }, delay);
}

function bindDraftValidationInputs() {
  const ids = [
    "draftTitle",
    "draftText",
    "draftDeltaJson",
    ...currentDraft.choices.flatMap((_, index) => [
      `draft_choice_${index}_type`,
      `draft_choice_${index}_label`
    ])
  ];

  for (const id of ids) {
    const element = document.getElementById(id);
    if (!element || element.dataset.validationBound === "1") {
      continue;
    }

    const handler = () => {
      if (id === "draftText") {
        updateDraftWordCount();
      }
      scheduleDraftValidation();
    };

    element.addEventListener("input", handler);
    element.addEventListener("change", handler);
    element.dataset.validationBound = "1";
  }
}

function collectAiChoices() {
  return currentDraft.choices.map(
    (choice, index) => ({
      id: choice.id,

      action_type:
        document.getElementById(`draft_choice_${index}_type`).value,

      label:
        document.getElementById(`draft_choice_${index}_label`).value.trim()
    })
  );
}

function readDeltaJson(id) {
  const raw =
    document
      .getElementById(id)
      .value
      .trim();

  if (!raw) {
    return emptyDelta();
  }

  try {
    return {
      ...emptyDelta(),
      ...JSON.parse(raw)
    };
  } catch (error) {
    throw new Error(
      "Memory changes contain invalid JSON."
    );
  }
}
async function saveAiDraft() {
  if (
    !currentDraft ||
    !currentGenerationRequest ||
    !currentGenerationMetrics ||
    currentGenerationLogId == null
  ) {
    alert("Generate a draft before saving.");
    return;
  }

  const button = document.getElementById("saveAiDraftBtn");
  button.disabled = true;

  try {
    const validation = await validateCurrentAiDraft();

    if (!validation || !validation.can_save) {
      document
        .getElementById("draftValidationWarning")
        .scrollIntoView({ behavior: "smooth", block: "center" });
      return;
    }

    if (
      validation.has_warnings &&
      !confirm(
        "This draft still has non-blocking warnings. " +
        "You have reviewed the highlighted fields. Save it anyway?"
      )
    ) {
      return;
    }

    const title = document.getElementById("draftTitle").value.trim();
    const storyText = document.getElementById("draftText").value.trim();
    const choices = collectAiChoices();
    const appliedDelta = readDeltaJson("draftDeltaJson");

    const result = await apiRequest("/nodes", {
      method: "POST",
      body: JSON.stringify({
        project_id: currentGenerationRequest.project_id,
        source_branch_id: currentGenerationRequest.branch_id,
        parent_node_id: currentGenerationRequest.parent_node_id,
        selected_choice_id: currentGenerationRequest.selected_choice_id,
        branch_name:
          document.getElementById("aiBranchName").value.trim() || null,
        force_fork: document.getElementById("aiForceFork").checked,
        title,
        story_text: storyText,
        applied_delta: appliedDelta,
        choices,
        authoring_mode: "ai",
        interaction_type: currentGenerationRequest.interaction_type,
        interaction_text: currentGenerationRequest.player_input,
        generated_by_model: currentGenerationMetrics.model,
        generation_log_id: currentGenerationLogId,
        review_override:
          !currentGenerationValidationPass || validation.has_warnings,
        scene_image_log_id: draftSceneImageLogId
      })
    });

    appState = result.state;
    activeBranchId = result.branch.id;
    selectedNodeId = result.node.id;
    selectedChoiceId = null;

    clearDraft();
    document.getElementById("playerInput").value = "";
    document.getElementById("gmInstruction").value = "";
    transcriptionLogsByTarget.delete("playerInput");
    transcriptionLogsByTarget.delete("gmInstruction");
    setDictationStatus("playerInput", dictationDefaultStatus("playerInput"));
    setDictationStatus("gmInstruction", dictationDefaultStatus("gmInstruction"));
    document.getElementById("aiBranchName").value = "";
    document.getElementById("aiForceFork").checked = false;

    await loadProjects();
    renderWorkspace();
    await renderObjectives();

    alert(
      result.fork_created
        ? "AI scene saved as a new branch."
        : "AI scene saved as the next node."
    );
  } catch (error) {
    alert(`Could not save AI scene: ${error.message}`);
  } finally {
    button.disabled = Boolean(
      currentDraftValidation?.blocking_count
    );
  }
}

function clearDraft() {
  currentDraft = null;
  currentGenerationLogId = null;
  currentGenerationMetrics = null;
  currentGenerationRequest = null;
  currentGenerationContext = null;
  currentGenerationValidationPass = true;
  currentGenerationIssues = [];
  currentDraftValidation = null;

  if (draftValidationTimerId !== null) {
    clearTimeout(draftValidationTimerId);
    draftValidationTimerId = null;
  }

  const contextView =
    document.getElementById("aiContextJson");

  if (contextView) {contextView.textContent = "Generate a draft to inspect its context.";}

  document
    .getElementById("aiDraftWorkspace")
    .classList
    .add("hidden");

  document.getElementById("generationStatus").textContent = "";
  clearGeneratedSceneImage("ai");
  const draftPrompt = document.getElementById("draftImagePrompt");
  if (draftPrompt) {
    draftPrompt.value = "";
  }

}

async function loadStoryElementCatalog() {
  storyElementCatalogData =
    await apiRequest("/story-elements");

  renderStoryElementPackFilter();
  renderStoryElementLibrary();
  renderStoryElementManagerFilters();
  renderStoryElementManagerList();

  if (editingStoryElementId) {
    const selected =
      storyElementCatalogData.find(
        (element) =>
          element.id === editingStoryElementId
      );

    if (selected) {
      openStoryElementEditor(selected);
    } else {
      clearStoryElementEditor();
    }
  }
}


function renderStoryElementPackFilter() {
  const select =
    document.getElementById(
      "setupStoryElementPackFilter"
    );

  if (!select) return;

  const previous = select.value;

  const packs = [
    ...new Set(
      storyElementCatalogData.map(
        (element) => element.pack
      )
    )
  ].sort();

  select.innerHTML = [
    `<option value="">All packs</option>`,
    ...packs.map(
      (pack) =>
        `<option value="${escapeHtml(pack)}">
          ${escapeHtml(pack)}
        </option>`
    )
  ].join("");

  if (packs.includes(previous)) {
    select.value = previous;
  }
}


function storyElementDetailsHtml(element) {
  const entries =
    Object.entries(
      element.details || {}
    );

  if (!entries.length) {
    return `
      <p class="subtle">
        No additional structured details.
      </p>
    `;
  }

  return `
    <dl class="story-element-detail-list">
      ${entries.map(([key, value]) => `
        <div>
          <dt>
            ${escapeHtml(
              key
                .replaceAll("_", " ")
                .replace(/\b\w/g, (letter) =>
                  letter.toUpperCase()
                )
            )}
          </dt>
          <dd>
            ${escapeHtml(
              Array.isArray(value)
                ? value.join(" · ")
                : value
            )}
          </dd>
        </div>
      `).join("")}
    </dl>
  `;
}


function renderStoryElementLibrary() {
  const container =
    document.getElementById(
      "storyElementCatalog"
    );

  if (!container) return;

  const mode =
    document.getElementById(
      "setupStoryElementMode"
    )?.value || "guided";

  const pack =
    document.getElementById(
      "setupStoryElementPackFilter"
    )?.value || "";

  const elements =
    storyElementCatalogData.filter(
      (element) =>
        !pack ||
        element.pack === pack
    );

  if (!elements.length) {
    container.innerHTML =
      `<p class="subtle">
        No story elements match this pack.
      </p>`;
    return;
  }

  container.innerHTML =
    elements.map(
      (element) => {
        const preference =
          draftStoryElementSelections
            .get(element.id);

        const enabled =
          Boolean(preference);

        return `
          <article class="story-element-row">

            <div class="story-element-selection-row">

              <label class="story-element-main">

                <input
                  type="checkbox"
                  data-story-element-enabled="${escapeHtml(element.id)}"
                  ${enabled ? "checked" : ""}
                  ${mode === "off" ? "disabled" : ""}
                />

                <span>
                  <strong>
                    ${escapeHtml(element.name)}
                  </strong>

                  <small>
                    ${escapeHtml(element.pack)}
                    ·
                    ${escapeHtml(element.category)}
                    ·
                    ${
                      element.is_builtin
                        ? "Built-in"
                        : "Custom"
                    }
                    ${
                      element.threat
                        ? ` · ${escapeHtml(element.threat)} threat`
                        : ""
                    }
                  </small>

                  <span class="story-element-description">
                    ${escapeHtml(element.description || "")}
                  </span>
                </span>

              </label>

              <select
                data-story-element-preference="${escapeHtml(element.id)}"
                ${!enabled || mode === "off" ? "disabled" : ""}
              >
                <option
                  value="available"
                  ${preference !== "preferred" ? "selected" : ""}
                >
                  Available
                </option>
                <option
                  value="preferred"
                  ${preference === "preferred" ? "selected" : ""}
                >
                  Preferred
                </option>
              </select>

            </div>

            <details class="story-element-inline-details">
              <summary>View all details</summary>
              ${storyElementDetailsHtml(element)}
            </details>

          </article>
        `;
      }
    ).join("");

  container
    .querySelectorAll(
      "[data-story-element-enabled]"
    )
    .forEach(
      (checkbox) => {
        checkbox.addEventListener(
          "change",
          () => {
            const id =
              checkbox.dataset
                .storyElementEnabled;

            if (checkbox.checked) {
              draftStoryElementSelections
                .set(
                  id,
                  "available"
                );
            } else {
              draftStoryElementSelections
                .delete(id);
            }

            renderStoryElementLibrary();
          }
        );
      }
    );

  container
    .querySelectorAll(
      "[data-story-element-preference]"
    )
    .forEach(
      (select) => {
        select.addEventListener(
          "change",
          () => {
            const id =
              select.dataset
                .storyElementPreference;

            if (
              draftStoryElementSelections
                .has(id)
            ) {
              draftStoryElementSelections
                .set(
                  id,
                  select.value
                );
            }
          }
        );
      }
    );
}


function renderStoryElementManagerFilters() {
  const packSelect =
    document.getElementById(
      "storyElementManagerPackFilter"
    );

  if (!packSelect) return;

  const previous =
    packSelect.value;

  const packs = [
    ...new Set(
      storyElementCatalogData.map(
        (element) => element.pack
      )
    )
  ].sort();

  packSelect.innerHTML = [
    `<option value="">All packs</option>`,
    ...packs.map(
      (pack) =>
        `<option value="${escapeHtml(pack)}">
          ${escapeHtml(pack)}
        </option>`
    )
  ].join("");

  if (packs.includes(previous)) {
    packSelect.value = previous;
  }
}


function renderStoryElementManagerList() {
  const container =
    document.getElementById(
      "storyElementManagerList"
    );

  if (!container) return;

  const query =
    (
      document.getElementById(
        "storyElementSearch"
      )?.value || ""
    ).trim().toLowerCase();

  const pack =
    document.getElementById(
      "storyElementManagerPackFilter"
    )?.value || "";

  const category =
    document.getElementById(
      "storyElementManagerCategoryFilter"
    )?.value || "";

  const filtered =
    storyElementCatalogData.filter(
      (element) => {
        const searchable = [
          element.name,
          element.pack,
          element.category,
          ...(element.tags || [])
        ].join(" ").toLowerCase();

        return (
          (!query || searchable.includes(query))
          &&
          (!pack || element.pack === pack)
          &&
          (!category || element.category === category)
        );
      }
    );

  document.getElementById(
    "storyElementCount"
  ).textContent =
    `${filtered.length} element${
      filtered.length === 1 ? "" : "s"
    }`;

  if (!filtered.length) {
    container.innerHTML =
      `<p class="subtle">
        No story elements match these filters.
      </p>`;
    return;
  }

  container.innerHTML =
    filtered.map(
      (element) => `
        <button
          type="button"
          class="story-element-manager-item ${
            element.id === editingStoryElementId
              ? "active"
              : ""
          }"
          data-manage-story-element="${escapeHtml(element.id)}"
        >
          <span>
            <strong>
              ${escapeHtml(element.name)}
            </strong>
            <small>
              ${escapeHtml(element.pack)}
              ·
              ${escapeHtml(element.category)}
            </small>
          </span>

          <span class="library-source-badge ${
            element.is_builtin
              ? "builtin"
              : "custom"
          }">
            ${
              element.is_builtin
                ? "Built-in"
                : "Custom"
            }
          </span>
        </button>
      `
    ).join("");

  container
    .querySelectorAll(
      "[data-manage-story-element]"
    )
    .forEach(
      (button) => {
        button.addEventListener(
          "click",
          () => {
            const element =
              storyElementCatalogData.find(
                (item) =>
                  item.id ===
                  button.dataset
                    .manageStoryElement
              );

            if (element) {
              openStoryElementEditor(
                element
              );
            }
          }
        );
      }
    );
}


function detailFieldConfiguration(
  category,
  details = {}
) {
  const configured = [
    ...(
      STORY_ELEMENT_DETAIL_FIELDS[
        category
      ] || []
    )
  ];

  const known =
    new Set(
      configured.map(
        (field) => field.key
      )
    );

  for (
    const [key, value]
    of Object.entries(details)
  ) {
    if (known.has(key)) {
      continue;
    }

    configured.push({
      key,
      label:
        key
          .replaceAll("_", " ")
          .replace(
            /\b\w/g,
            (letter) =>
              letter.toUpperCase()
          ),
      kind:
        Array.isArray(value)
          ? "list"
          : "text"
    });
  }

  return configured;
}


function renderStoryElementDetailFields(
  category,
  details = {}
) {
  const container =
    document.getElementById(
      "libraryElementDetailFields"
    );

  if (!container) return;

  const fields =
    detailFieldConfiguration(
      category,
      details
    );

  if (!fields.length) {
    container.innerHTML =
      `<p class="subtle span-two">
        No additional fields are configured for this category.
      </p>`;
    return;
  }

  container.innerHTML =
    fields.map(
      (field) => {
        const value =
          details[field.key];

        const displayValue =
          Array.isArray(value)
            ? value.join("\n")
            : (value || "");

        const help =
          field.kind === "list"
            ? "One entry per line."
            : "";

        return `
          <div class="${
            field.kind === "text"
              ? "span-two"
              : ""
          }">
            <label
              for="library_detail_${escapeHtml(field.key)}"
            >
              ${escapeHtml(field.label)}
            </label>

            <textarea
              id="library_detail_${escapeHtml(field.key)}"
              data-element-detail-key="${escapeHtml(field.key)}"
              data-element-detail-kind="${escapeHtml(field.kind)}"
              placeholder="${escapeHtml(help)}"
            >${escapeHtml(displayValue)}</textarea>
          </div>
        `;
      }
    ).join("");
}


function currentEditorDetails() {
  const details = {};

  document
    .querySelectorAll(
      "[data-element-detail-key]"
    )
    .forEach(
      (field) => {
        const value =
          field.value.trim();

        if (!value) {
          return;
        }

        details[
          field.dataset.elementDetailKey
        ] =
          field.dataset
            .elementDetailKind === "list"
            ? lines(value)
            : value;
      }
    );

  return details;
}


function openStoryElementEditor(
  element
) {
  editingStoryElementId =
    element.id;

  editingStoryElementSnapshot =
    structuredClone(element);

  document.getElementById(
    "storyElementEditorForm"
  ).classList.remove("hidden");

  document.getElementById(
    "storyElementEditorEmpty"
  ).classList.add("hidden");

  document.getElementById(
    "storyElementEditorTitle"
  ).textContent =
    element.name;

  document.getElementById(
    "storyElementEditorSource"
  ).textContent =
    element.is_builtin
      ? "Built-in element"
      : "User-created element";

  document.getElementById(
    "libraryElementName"
  ).value =
    element.name || "";

  document.getElementById(
    "libraryElementPack"
  ).value =
    element.pack || "Custom";

  document.getElementById(
    "libraryElementCategory"
  ).value =
    element.category || "creature";

  document.getElementById(
    "libraryElementThreat"
  ).value =
    element.threat || "";

  document.getElementById(
    "libraryElementTags"
  ).value =
    (element.tags || []).join(", ");

  document.getElementById(
    "libraryElementDescription"
  ).value =
    element.description || "";

  renderStoryElementDetailFields(
    element.category || "creature",
    element.details || {}
  );

  document.getElementById(
    "resetStoryElementBtn"
  ).classList.toggle(
    "hidden",
    !element.is_builtin
  );

  document.getElementById(
    "deleteStoryElementBtn"
  ).classList.remove("hidden");

  document.getElementById(
    "deleteStoryElementBtn"
  ).textContent =
    element.is_builtin
      ? "Hide built-in"
      : "Delete custom element";

  renderStoryElementManagerList();
}


function clearStoryElementEditor() {
  editingStoryElementId = null;
  editingStoryElementSnapshot = null;

  document.getElementById(
    "storyElementEditorForm"
  )?.classList.add("hidden");

  document.getElementById(
    "storyElementEditorEmpty"
  )?.classList.remove("hidden");

  document.getElementById(
    "storyElementEditorTitle"
  ).textContent =
    "Story element editor";

  document.getElementById(
    "storyElementEditorSource"
  ).textContent =
    "Select an element";

  renderStoryElementManagerList();
}


function startNewStoryElement() {
  editingStoryElementId = null;

  editingStoryElementSnapshot = {
    id: null,
    pack: "Custom",
    category: "creature",
    name: "",
    tags: [],
    threat: null,
    description: "",
    details: {},
    is_builtin: false
  };

  document.getElementById(
    "storyElementEditorForm"
  ).classList.remove("hidden");

  document.getElementById(
    "storyElementEditorEmpty"
  ).classList.add("hidden");

  document.getElementById(
    "storyElementEditorTitle"
  ).textContent =
    "New story element";

  document.getElementById(
    "storyElementEditorSource"
  ).textContent =
    "User-created element";

  document.getElementById(
    "libraryElementName"
  ).value = "";

  document.getElementById(
    "libraryElementPack"
  ).value = "Custom";

  document.getElementById(
    "libraryElementCategory"
  ).value = "creature";

  document.getElementById(
    "libraryElementThreat"
  ).value = "";

  document.getElementById(
    "libraryElementTags"
  ).value = "";

  document.getElementById(
    "libraryElementDescription"
  ).value = "";

  renderStoryElementDetailFields(
    "creature",
    {}
  );

  document.getElementById(
    "resetStoryElementBtn"
  ).classList.add("hidden");

  document.getElementById(
    "deleteStoryElementBtn"
  ).classList.add("hidden");

  renderStoryElementManagerList();
}


function storyElementEditorPayload() {
  const name =
    document.getElementById(
      "libraryElementName"
    ).value.trim();

  const pack =
    document.getElementById(
      "libraryElementPack"
    ).value.trim();

  const description =
    document.getElementById(
      "libraryElementDescription"
    ).value.trim();

  if (!name) {
    throw new Error(
      "Story element name is required."
    );
  }

  if (!pack) {
    throw new Error(
      "Story element pack is required."
    );
  }

  return {
    pack,
    category:
      document.getElementById(
        "libraryElementCategory"
      ).value,
    name,
    tags:
      commaList(
        document.getElementById(
          "libraryElementTags"
        ).value
      ),
    threat:
      document.getElementById(
        "libraryElementThreat"
      ).value || null,
    description,
    details:
      currentEditorDetails()
  };
}


async function saveStoryElement() {
  const button =
    document.getElementById(
      "saveStoryElementBtn"
    );

  button.disabled = true;

  try {
    const payload =
      storyElementEditorPayload();

    const saved =
      editingStoryElementId
        ? await apiRequest(
            `/story-elements/${encodeURIComponent(editingStoryElementId)}`,
            {
              method: "PATCH",
              body:
                JSON.stringify(payload)
            }
          )
        : await apiRequest(
            "/story-elements",
            {
              method: "POST",
              body:
                JSON.stringify(payload)
            }
          );

    editingStoryElementId =
      saved.id;

    await loadStoryElementCatalog();

    alert(
      "Story element saved."
    );
  } catch (error) {
    alert(
      `Could not save story element: ${error.message}`
    );
  } finally {
    button.disabled = false;
  }
}


async function deleteStoryElement() {
  if (!editingStoryElementId) {
    return;
  }

  const element =
    storyElementCatalogData.find(
      (item) =>
        item.id === editingStoryElementId
    );

  if (!element) {
    return;
  }

  const message =
    element.is_builtin
      ? (
          `Hide the built-in element '${element.name}'? ` +
          "You can restore it later with Reset built-in defaults."
        )
      : (
          `Delete the custom element '${element.name}'?`
        );

  if (!confirm(message)) {
    return;
  }

  try {
    await apiRequest(
      `/story-elements/${encodeURIComponent(element.id)}`,
      {
        method: "DELETE"
      }
    );

    clearStoryElementEditor();
    await loadStoryElementCatalog();

  } catch (error) {
    alert(
      `Could not delete story element: ${error.message}`
    );
  }
}


async function resetSelectedStoryElement() {
  if (!editingStoryElementId) {
    return;
  }

  const element =
    storyElementCatalogData.find(
      (item) =>
        item.id === editingStoryElementId
    );

  if (
    !element ||
    !element.is_builtin
  ) {
    return;
  }

  if (
    !confirm(
      `Reset '${element.name}' to its original bundled values?`
    )
  ) {
    return;
  }

  try {
    await apiRequest(
      `/story-elements/${encodeURIComponent(element.id)}/reset`,
      {
        method: "POST"
      }
    );

    await loadStoryElementCatalog();

  } catch (error) {
    alert(
      `Could not reset story element: ${error.message}`
    );
  }
}


async function resetAllStoryElements() {
  if (
    !confirm(
      "Reset all built-in story elements to their original bundled values? Custom elements will not be changed."
    )
  ) {
    return;
  }

  try {
    await apiRequest(
      "/story-elements/reset-defaults",
      {
        method: "POST"
      }
    );

    await loadStoryElementCatalog();

    alert(
      "Built-in Story Elements restored."
    );

  } catch (error) {
    alert(
      `Could not reset Story Elements: ${error.message}`
    );
  }
}


async function openStoryElementsManager(
  returnView
) {
  storyElementsReturnView =
    returnView || (
      currentProjectId
        ? "workspace"
        : "home"
    );

  document.getElementById(
    "homeView"
  ).classList.add("hidden");

  document.getElementById(
    "workspaceView"
  ).classList.add("hidden");

  document.getElementById(
    "storyElementsView"
  ).classList.remove("hidden");

  try {
    await loadStoryElementCatalog();
  } catch (error) {
    alert(
      `Could not load Story Elements: ${error.message}`
    );
  }
}


function closeStoryElementsManager() {
  document.getElementById(
    "storyElementsView"
  ).classList.add("hidden");

  if (
    storyElementsReturnView === "workspace"
    &&
    currentProjectId
    &&
    appState
  ) {
    document.getElementById(
      "workspaceView"
    ).classList.remove("hidden");

    renderWorkspace();
  } else {
    document.getElementById(
      "homeView"
    ).classList.remove("hidden");

    loadProjects().catch(console.error);
  }
}


function projectPayload() {
  return {
    title: document.getElementById("setupTitle").value.trim(),
    genre: document.getElementById("setupGenre").value.trim() || "Fantasy",
    tone: document.getElementById("setupTone").value.trim() || "Adventurous",
    world_summary: document.getElementById("setupWorld").value.trim(),
    main_objective: document.getElementById("setupObjective").value.trim(),
    story_element_mode: document.getElementById("setupStoryElementMode").value,
    story_elements: [...draftStoryElementSelections.entries()].map(
      ([elementId, preference]) => ({
        element_id: elementId,
        preference
      })
    ),
    starting_scene_title: document.getElementById("setupSceneTitle").value.trim() || "Starting scene",
    starting_scene_text: document.getElementById("setupSceneText").value.trim(),
    initial_memory: {
      location: document.getElementById("setupLocation").value.trim() || "Unknown",
      active_characters: commaList(document.getElementById("setupCharacters").value),
      inventory: commaList(document.getElementById("setupInventory").value),
      relationships: relationshipsFromText(document.getElementById("setupRelationships").value),
      goals: lines(document.getElementById("setupGoals").value),
      unresolved_clues: lines(document.getElementById("setupClues").value),
      decisions: lines(document.getElementById("setupDecisions").value),
      threat: document.getElementById("setupThreat").value
    }
  };
}

function openCreateProjectModal() {
  editingProject = false;
  document.getElementById("projectModalTitle").textContent = "Create a story";
  document.getElementById("settingsWarning").classList.add("hidden");
  for (const id of ["setupTitle", "setupObjective", "setupWorld", "setupSceneText", "setupCharacters", "setupInventory", "setupRelationships", "setupGoals", "setupClues", "setupDecisions"]) document.getElementById(id).value = "";
  document.getElementById("setupGenre").value = "Fantasy";
  document.getElementById("setupTone").value = "Adventurous";
  document.getElementById("setupSceneTitle").value = "Starting scene";
  document.getElementById("setupLocation").value = "Unknown";
  document.getElementById("setupThreat").value = "Low";
  document.getElementById("setupStoryElementMode").value = "guided";
  document.getElementById("setupStoryElementPackFilter").value = "";
  draftStoryElementSelections = new Map();
  renderStoryElementLibrary();
  openModal("projectModal");
}

function openProjectSettings() {
  const project = getProject();
  const root = appState.nodes.find((node) => !node.parent_node_id);
  if (!project || !root) return;
  editingProject = true;
  document.getElementById("projectModalTitle").textContent = "Story setup settings";
  document.getElementById("settingsWarning").classList.toggle("hidden", appState.nodes.length <= 1);
  document.getElementById("setupTitle").value = project.title;
  document.getElementById("setupGenre").value = project.genre;
  document.getElementById("setupTone").value = project.tone;
  document.getElementById("setupObjective").value = project.main_objective;
  document.getElementById("setupWorld").value = project.world_summary;
  document.getElementById("setupSceneTitle").value = root.title;
  document.getElementById("setupSceneText").value = root.story_text;
  document.getElementById("setupLocation").value = root.memory_snapshot.location || "Unknown";
  document.getElementById("setupCharacters").value = (root.memory_snapshot.active_characters || []).join(", ");
  document.getElementById("setupInventory").value = (root.memory_snapshot.inventory || []).join(", ");
  document.getElementById("setupRelationships").value = relationshipsToText(root.memory_snapshot.relationships || {});
  document.getElementById("setupGoals").value = (root.memory_snapshot.goals || []).join("\n");
  document.getElementById("setupClues").value = (root.memory_snapshot.unresolved_clues || []).join("\n");
  document.getElementById("setupDecisions").value = (root.memory_snapshot.decisions || []).join("\n");
  document.getElementById("setupThreat").value = root.memory_snapshot.threat || "Low";
  document.getElementById("setupStoryElementMode").value = project.story_element_mode || "guided";
  draftStoryElementSelections = new Map(
    (project.story_elements || []).map((selection) => [
      selection.element_id,
      selection.preference || "available"
    ])
  );
  renderStoryElementLibrary();
  openModal("projectModal");
}

async function saveProject() {
  const payload = projectPayload();
  if (!payload.title || payload.starting_scene_text.length < 20) return alert("Enter a title and a starting scene of at least 20 characters.");
  const button = document.getElementById("saveProjectBtn");
  button.disabled = true;
  try {
    if (editingProject) {
      appState = await apiRequest(
        `/projects/${encodeURIComponent(currentProjectId)}`,
        {
          method: "PATCH",
          body: JSON.stringify(payload)
        }
      );

      clearDraft();

      closeModal("projectModal");
      await loadProjects();
      renderWorkspace();
    }
    else {
      const state = await apiRequest("/projects", { method: "POST", body: JSON.stringify(payload) });
      closeModal("projectModal");
      await loadProjects();
      await openProject(state.project.id);
    }
  } catch (error) {
    alert(`Could not save story: ${error.message}`);
  } finally {
    button.disabled = false;
  }
}

async function viewFullStory() {
  if (!selectedNodeId) return;
  try {
    const result = await apiRequest(`/nodes/${encodeURIComponent(selectedNodeId)}/path`);
    document.getElementById("fullStoryPathLabel").textContent = `${result.project.title} · ${result.path.length} scenes`;
    document.getElementById("fullStoryContent").innerHTML = result.path.map((node, index) => `
      <section class="full-story-node">
        ${node.source_choice_label ? `<div class="path-choice-label">Choice: ${escapeHtml(node.source_choice_label)}</div>` : ""}
        <h3>${index + 1}. ${escapeHtml(node.title)}</h3>
        <p>${escapeHtml(node.story_text)}</p>
      </section>
    `).join("");
    openModal("fullStoryModal");
  } catch (error) {
    alert(`Could not load full story: ${error.message}`);
  }
}

function openEditNode() {
  const node = getNode();
  if (!node) return;
  document.getElementById("editNodeTitle").value = node.title;
  document.getElementById("editNodeText").value = node.story_text;
  document.getElementById("editChoicesEditor").innerHTML = (node.choices || []).map((choice, index) => `
    <article class="choice-editor">
      <h4>Choice ${index + 1}</h4>
      <label for="edit_choice_${index}_type">Type</label>
      <select id="edit_choice_${index}_type">${["do", "ask", "say", "continue"].map((type) => `<option value="${type}" ${choice.action_type === type ? "selected" : ""}>${type.toUpperCase()}</option>`).join("")}</select>
      <label for="edit_choice_${index}_label">Choice text</label><input id="edit_choice_${index}_label" value="${escapeHtml(choice.label)}" />
      
    </article>
  `).join("") || `<p class="subtle">This node has no prepared choices.</p>`;
  document.getElementById("editDeltaEditor").innerHTML =
    deltaEditorHtml("edit", node.state_delta || {});
  openModal("editNodeModal");
}

async function saveNodeEdit() {
  const node = getNode();
  if (!node) return;

  const title = document.getElementById("editNodeTitle").value.trim();
  const storyText = document.getElementById("editNodeText").value.trim();
  const choices = (node.choices || []).map(
      (choice, index) => ({

        id: choice.id,

        action_type:
          document.getElementById(`edit_choice_${index}_type`).value,

        label:
          document.getElementById(`edit_choice_${index}_label`).value.trim()
      })
    );

  if (!title || !storyText) {
    return alert("Enter a node title and scene text.");
  }
  if (choices.some((choice) => !choice.label)) {
    return alert("Every existing choice must have a label.");
  }

  const button = document.getElementById("saveNodeEditBtn");
  button.disabled = true;

  try {
    const result = await apiRequest(`/nodes/${encodeURIComponent(node.id)}`, {
      method: "PATCH",
      body: JSON.stringify({
        title,
        story_text: storyText,
        choices,
        applied_delta: readDelta("edit")
      })
    });
    appState = result.state;
    clearDraft();
    closeModal("editNodeModal");
    await loadProjects();
    renderWorkspace();
    await renderObjectives();
    } catch (error) {
        alert(
          `Could not update node: ${error.message}`
        );
    } finally {
        button.disabled = false;
    }
}

async function deleteSelectedNode() {
  const node = getNode();
  if (!node) return;
  if (!confirm(`Delete '${node.title}'? Only a leaf scene can be deleted.`)) return;
  try {
    const result = await apiRequest(`/nodes/${encodeURIComponent(node.id)}`, { method: "DELETE" });
    appState = result.state;
    if (!getBranch(activeBranchId)) activeBranchId = appState.branches[0]?.id || null;
    selectedNodeId = getBranch()?.head_node_id || appState.nodes[0]?.id || null;
    selectedChoiceId = null;
    clearDraft();
    await loadProjects();
    renderWorkspace();
    await renderObjectives();
    alert("Node moved to Trash. It can be restored later.");
  } catch (error) {
    alert(`Could not delete node: ${error.message}`);
  }
}

async function deleteActiveBranch() {
  const branch = getBranch(activeBranchId);
  if (!branch) return;
  if (!branch.parent_branch_id) {
    return alert("The main branch cannot be deleted.");
  }

  if (!confirm(
    `Move '${branch.name}' and all branches forked from it to Trash?`
  )) return;

  try {
    const result = await apiRequest(
      `/branches/${encodeURIComponent(branch.id)}`,
      { method: "DELETE" }
    );

    appState = result.state;
    activeBranchId = getBranch(result.parent_branch_id)?.id
      || appState.branches[0]?.id
      || null;
    selectedNodeId = getBranch(activeBranchId)?.head_node_id
      || appState.nodes[0]?.id
      || null;
    selectedChoiceId = null;
    expandedMapNodeId = null;
    clearDraft();
    await loadProjects();
    renderWorkspace();
    await renderObjectives();
    alert("Branch moved to Trash. It can be restored later.");
  } catch (error) {
    alert(`Could not delete branch: ${error.message}`);
  }
}

async function openTrash() {
  if (!currentProjectId) return;
  const container = document.getElementById("trashContent");
  container.innerHTML = `<p class="subtle">Loading deleted items…</p>`;
  openModal("trashModal");

  try {
    const items = await apiRequest(
      `/projects/${encodeURIComponent(currentProjectId)}/trash`
    );

    if (!items.length) {
      container.innerHTML = `
        <div class="empty-home">
          <h3>Trash is empty</h3>
          <p>Deleted leaf nodes and forked branches will appear here.</p>
        </div>
      `;
      return;
    }

    container.innerHTML = items.map((item) => `
      <article class="trash-item">
        <div class="trash-item-heading">
          <div>
            <p class="eyebrow">${escapeHtml(item.kind)}</p>
            <h3>${escapeHtml(item.title)}</h3>
            <p>
              ${item.node_count} node${item.node_count === 1 ? "" : "s"}
              · ${item.branch_count} branch${item.branch_count === 1 ? "" : "es"}
              · deleted ${escapeHtml(formatDate(item.deleted_at))}
            </p>
          </div>
          <button
            type="button"
            class="secondary-button"
            data-restore-batch="${escapeHtml(item.batch_id)}"
          >Restore</button>
        </div>
      </article>
    `).join("");

    container.querySelectorAll("[data-restore-batch]").forEach((button) => {
      button.addEventListener("click", async () => {
        button.disabled = true;
        try {
          const result = await apiRequest(
            `/trash/${encodeURIComponent(button.dataset.restoreBatch)}/restore`,
            { method: "POST" }
          );
          appState = result.state;
          if (!getBranch(activeBranchId)) {
            activeBranchId = appState.branches[0]?.id || null;
          }
          selectedNodeId = getBranch(activeBranchId)?.head_node_id
            || appState.nodes[0]?.id
            || null;
          selectedChoiceId = null;
          expandedMapNodeId = null;
          await loadProjects();
          renderWorkspace();
          await renderObjectives();
          await openTrash();
        } catch (error) {
          alert(`Could not restore item: ${error.message}`);
        } finally {
          button.disabled = false;
        }
      });
    });
  } catch (error) {
    container.innerHTML = `
      <p class="warning-text">${escapeHtml(error.message)}</p>
    `;
  }
}

function renderCompareSelectors() {
  const options = appState.branches.map((branch) => `<option value="${escapeHtml(branch.id)}">${escapeHtml(branch.name)}</option>`).join("");
  const left = document.getElementById("leftBranchSelect");
  const right = document.getElementById("rightBranchSelect");
  const oldLeft = left.value;
  const oldRight = right.value;
  left.innerHTML = options;
  right.innerHTML = options;
  left.value = getBranch(oldLeft)?.id || appState.branches[0]?.id || "";
  right.value = getBranch(oldRight)?.id || appState.branches.at(-1)?.id || "";
}

async function compareBranches() {
  const leftId = document.getElementById("leftBranchSelect").value;
  const rightId = document.getElementById("rightBranchSelect").value;
  if (!leftId || !rightId) return;
  try {
    const result = await apiRequest(`/compare?left_branch_id=${encodeURIComponent(leftId)}&right_branch_id=${encodeURIComponent(rightId)}`);
    const rows = Object.entries(result.memory_diff).map(([field, diff]) => `
      <tr class="${diff.different ? "different" : ""}"><th>${escapeHtml(field.replaceAll("_", " "))}</th><td>${escapeHtml(formatValue(diff.left))}</td><td>${escapeHtml(formatValue(diff.right))}</td></tr>
    `).join("");
    document.getElementById("compareContent").innerHTML = `
      <p><strong>Common ancestor:</strong> ${escapeHtml(result.common_ancestor?.title || "None")}</p>
      <div class="compare-grid">
        <article class="compare-column"><h3>${escapeHtml(result.left_branch.name)}</h3><ol>${result.left_unique_nodes.map((node) => `<li>${escapeHtml(node.title)}</li>`).join("") || "<li>No unique scenes</li>"}</ol></article>
        <article class="compare-column"><h3>${escapeHtml(result.right_branch.name)}</h3><ol>${result.right_unique_nodes.map((node) => `<li>${escapeHtml(node.title)}</li>`).join("") || "<li>No unique scenes</li>"}</ol></article>
      </div>
      <table class="diff-table"><thead><tr><th>Memory field</th><th>${escapeHtml(result.left_branch.name)}</th><th>${escapeHtml(result.right_branch.name)}</th></tr></thead><tbody>${rows}</tbody></table>
    `;
  } catch (error) {
    document.getElementById("compareContent").innerHTML = `<p class="warning-text">${escapeHtml(error.message)}</p>`;
  }
}

async function renderObjectives() {
  const container =
    document.getElementById(
      "objectivePanel"
    );

  if (!currentProjectId) {
    return;
  }

  try {
    const data = await apiRequest(
      `/projects/${encodeURIComponent(
        currentProjectId
      )}/objectives`
    );

    const o1 = data.objective_1;
    const o2 = data.objective_2;
    const o3 = data.objective_3;

    /*
     * Objective 1 is only met if every stated
     * success criterion represented here passes.
     */
    const objective1Met =
      o1.two_or_more_branches &&
      o1.branches_share_a_starting_point &&
      o1.all_ai_nodes_have_three_choices &&
      o1.latency_target_met;

    /*
     * Objective 2 cannot yet be declared passed
     * from schema completeness alone.
     *
     * The formal continuity experiment must supply
     * fact-preservation and contradiction results.
     */


    const averageGeneration = o1.average_generation_seconds ?? "not measured";
    container.innerHTML = `
      <article
        class="objective-card ${
          objective1Met
            ? "pass"
            : "warn"
        }"
      >
        <h3>
          Objective 1 · Editable branches
        </h3>

        <p>
          ${o1.manual_scene_count}
          manually written and
          ${o1.ai_scene_count}
          AI-generated scenes across
          ${o1.branch_count} branches.
          Average generation:
          ${averageGeneration}s.
          Target: under
          ${o1.target_generation_seconds}s.
        </p>
      </article>

      <article
        class="objective-card warn"
      >
        <h3>
          Objective 2 · Branch continuity
        </h3>

        <p>
          ${o2.node_memory_snapshot_count}
          node-level memory snapshots are available. Formal continuity evaluation is still required:
          target ≥85% facts preserved, 0 critical contradictions and
          ≤${o2.continuity_errors_per_branch_target}
          continuity errors per branch.
        </p>
      </article>

      <article
        class="objective-card ${
          o3.target_met
            ? "pass"
            : "warn"
        }"
      >
        <h3>
          Objective 3 · Realised consequences
        </h3>

        <p>
          ${o3.selected_choices_with_state_change}
          of
          ${o3.selected_choice_count}
          selected choices produced a stored state change
          (${(o3.state_changing_choice_ratio * 100).toFixed(1)}% target ≥80%).
        </p>
      </article>
    `;

  } catch (error) {
    container.innerHTML = `
      <article class="objective-card warn">
        <h3>
          Objective evidence unavailable
        </h3>

        <p>
          ${escapeHtml(error.message)}
        </p>
      </article>
    `;
  }
}

function bindEvents() {
  document.getElementById("newStoryHomeBtn").addEventListener("click", openCreateProjectModal);
  document.getElementById("homeStoryElementsBtn").addEventListener(
    "click",
    () => openStoryElementsManager("home")
  );
  document.getElementById("workspaceStoryElementsBtn").addEventListener(
    "click",
    () => openStoryElementsManager("workspace")
  );
  document.getElementById("backFromStoryElementsBtn").addEventListener(
    "click",
    closeStoryElementsManager
  );
  document.getElementById("newStoryElementBtn").addEventListener(
    "click",
    startNewStoryElement
  );
  document.getElementById("saveStoryElementBtn").addEventListener(
    "click",
    saveStoryElement
  );
  document.getElementById("deleteStoryElementBtn").addEventListener(
    "click",
    deleteStoryElement
  );
  document.getElementById("resetStoryElementBtn").addEventListener(
    "click",
    resetSelectedStoryElement
  );
  document.getElementById("resetAllStoryElementsBtn").addEventListener(
    "click",
    resetAllStoryElements
  );
  document.getElementById("storyElementSearch").addEventListener(
    "input",
    renderStoryElementManagerList
  );
  document.getElementById("storyElementManagerPackFilter").addEventListener(
    "change",
    renderStoryElementManagerList
  );
  document.getElementById("storyElementManagerCategoryFilter").addEventListener(
    "change",
    renderStoryElementManagerList
  );
  document.getElementById("libraryElementCategory").addEventListener(
    "change",
    () => {
      const existingDetails =
        currentEditorDetails();

      renderStoryElementDetailFields(
        document.getElementById(
          "libraryElementCategory"
        ).value,
        existingDetails
      );
    }
  );
  document.getElementById("blankStoryCard").addEventListener("click", openCreateProjectModal);
  document.getElementById("newStorySidebarBtn").addEventListener("click", openCreateProjectModal);
  document.getElementById("homeBtn").addEventListener("click", showHome);
  document.getElementById("refreshStoriesBtn").addEventListener("click",
    async () => {
      try {
        await loadProjects();
      } catch (error) {
        alert(
          "Could not refresh stories: " +
          error.message
        );
      }
    }
  );
  document.getElementById("storySearch").addEventListener("input", renderHomeStories);
  document.getElementById("openSettingsBtn").addEventListener("click", openProjectSettings);
  document.getElementById("saveProjectBtn").addEventListener("click", saveProject);
  document.getElementById("setupStoryElementPackFilter").addEventListener("change", renderStoryElementLibrary);
  document.getElementById("setupStoryElementMode").addEventListener("change", renderStoryElementLibrary);

  document.getElementById("viewFullStoryBtn").addEventListener("click", viewFullStory);
  document.getElementById("editNodeBtn").addEventListener("click", openEditNode);
  document.getElementById("saveNodeEditBtn").addEventListener("click", saveNodeEdit);
  document.getElementById("deleteNodeBtn").addEventListener("click", deleteSelectedNode);
  document.getElementById("deleteBranchBtn").addEventListener("click", deleteActiveBranch);
  document.getElementById("openTrashBtn").addEventListener("click", openTrash);
  document.getElementById("saveManualNodeBtn").addEventListener("click", saveManualNode);
  document.getElementById("manualSceneDictateBtn").addEventListener("click",() => toggleDictation("manualSceneText"));
  document.getElementById("playerInputDictateBtn").addEventListener("click", () => toggleDictation("playerInput"));
  document.getElementById("gmInstructionDictateBtn").addEventListener("click", () => toggleDictation("gmInstruction"));
  document.getElementById("generateDraftBtn").addEventListener("click", generateDraft);
  document.getElementById("regenerateDraftBtn").addEventListener("click", generateDraft);
  document.getElementById("saveAiDraftBtn").addEventListener("click", saveAiDraft);
  document.getElementById("compareBranchesBtn").addEventListener("click", () => { renderCompareSelectors(); openModal("compareModal"); });
  document.getElementById("runComparisonBtn").addEventListener("click", compareBranches);
  document.querySelectorAll(".composer-tab").forEach((button) => button.addEventListener("click", () => setComposerMode(button.dataset.composerMode)));
  document.getElementById("manualForceFork").addEventListener("change", renderComposerContext);
  document.getElementById("aiForceFork").addEventListener("change", renderComposerContext);
  document.querySelectorAll("[data-close-modal]").forEach((button) => button.addEventListener("click", () => closeModal(button.dataset.closeModal)));
  
  document.getElementById("manualGenerateImageBtn").addEventListener("click",() =>
        generateSceneImage("manual")
    );

  document.getElementById("draftGenerateImageBtn").addEventListener("click",() =>
      generateSceneImage("ai")
    );

  document.getElementById("copyMemoryJsonBtn").addEventListener("click",
    async () => {
      const value =
        document.getElementById("memoryJsonView").value;
      try {
        await navigator.clipboard.writeText(value);
      } catch (error) {
        alert(
          "Could not copy memory JSON."
        );
      }
    }
  );

  document.getElementById("regenerateSelectedNodeImageBtn").addEventListener("click",regenerateSelectedNodeImage);
}


async function start() {
  bindEvents();
  renderManualChoicesEditor();
  setComposerMode("write");
  await Promise.all([refreshHealth(), loadProjects(), loadStoryElementCatalog()]);
}

start().catch((error) => {
  console.error(error);
  document.getElementById("homeServiceStatus").className = "status-pill fail";
  document.getElementById("homeServiceStatus").textContent = `Startup failed: ${error.message}`;
});


function clearGeneratedSceneImage(kind, message = "No image generated.") {
  if (kind === "manual") {
    manualSceneImageLogId = null;
  } else {
    draftSceneImageLogId = null;
  }

  const ui = getImageUi(kind);
  const status = document.getElementById(ui.statusId);
  const preview = document.getElementById(ui.previewId);

  if (status) {
    status.textContent = message;
  }

  if (preview) {
    preview.innerHTML = "";
  }
}


async function generateSceneImage(kind) {
  const manual = kind === "manual";
  const ui = getImageUi(kind);

  const title = document.getElementById(ui.titleId).value.trim();
  const storyText = document.getElementById(ui.textId).value.trim();
  const promptOverride = document.getElementById(ui.promptId).value.trim();
  const modelKey = document.getElementById(ui.modelId).value;

  if (!currentProjectId || !activeBranchId || !selectedNodeId) {
    alert("Select a story node and branch first.");
    return;
  }

  if (!title || !storyText) {
    alert("Enter a scene title and scene text before generating an image.");
    return;
  }

  if (!manual && !currentGenerationRequest) {
    alert("Generate an AI draft before generating its image.");
    return;
  }

  const button = document.getElementById(ui.buttonId);
  const status = document.getElementById(ui.statusId);
  const preview = document.getElementById(ui.previewId);

  button.disabled = true;
  status.textContent = "Generating image locally…";

  try {
    const appliedDelta = manual
      ? manualDelta()
      : readDeltaJson("draftDeltaJson");

    const branchId = manual
      ? activeBranchId
      : currentGenerationRequest.branch_id;

    const parentNodeId = manual
      ? selectedNodeId
      : currentGenerationRequest.parent_node_id;

    const result = await apiRequest("/generate-scene-image", {
      method: "POST",
      body: JSON.stringify({
        project_id: currentProjectId,
        branch_id: branchId,
        parent_node_id: parentNodeId,
        source_kind: kind,
        model_key: modelKey,
        title,
        story_text: storyText,
        applied_delta: appliedDelta,
        prompt_override: promptOverride,
        seed: 42,
      }),
    });

    if (manual) {
      manualSceneImageLogId = result.image_generation_log_id;
    } else {
      draftSceneImageLogId = result.image_generation_log_id;
    }

    preview.innerHTML = `
      <img
        src="${escapeHtml(result.image_url)}"
        alt="Generated scene illustration"
      />
      <small>
        ${escapeHtml(result.model_key)}
        · ${escapeHtml(result.device)}
        · ${Number(result.inference_seconds).toFixed(2)} s
      </small>
    `;

    status.textContent =
      `Image ready in ${Number(result.total_seconds).toFixed(2)} s. ` +
      "Review it before saving.";
  } catch (error) {
    clearGeneratedSceneImage(
      kind,
      `Image generation failed: ${error.message}`,
    );
  } finally {
    button.disabled = false;
  }
}



function getImageUi(kind) {
  const manual = kind === "manual";

  return {
    titleId: manual ? "manualTitle" : "draftTitle",
    textId: manual ? "manualSceneText" : "draftText",
    promptId: manual ? "manualImagePrompt" : "draftImagePrompt",
    modelId: manual ? "manualImageModel" : "draftImageModel",
    buttonId: manual ? "manualGenerateImageBtn" : "draftGenerateImageBtn",
    statusId: manual ? "manualImageStatus" : "draftImageStatus",
    previewId: manual ? "manualImagePreview" : "draftImagePreview",
  };
}


async function regenerateSelectedNodeImage() {
  const node = getNode();
  if (!node) {alert("Select a story node first.");
    return;
  }

  const button = document.getElementById("regenerateSelectedNodeImageBtn");

  const status = document.getElementById("selectedNodeImageStatus");
  const promptOverride = document.getElementById("selectedNodeImagePrompt").value.trim();
  const modelKey = document.getElementById("selectedNodeImageModel").value;
  button.disabled = true;
  status.textContent = node.scene_image
      ? "Regenerating image locally…"
      : "Generating image locally…";

  try {
    const result =
      await apiRequest(
        `/nodes/${encodeURIComponent(node.id)}/scene-image`,
        {method: "POST",
          body: JSON.stringify({
            model_key:modelKey,
            prompt_override: promptOverride,
            seed: 42
          })
        }
      );

    appState = result.state;
    renderWorkspace();
    status.textContent =
      `Image saved to node in ${
        Number(result.image.total_seconds).toFixed(2)
      } s.`;

  } catch (error) {
    status.textContent =
      `Image generation failed: ${error.message}`;

  } finally {button.disabled = false;}
}