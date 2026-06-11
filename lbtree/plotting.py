"""
lbtree/plotting.py
==================
Interactive HTML tree visualisation for SCTree and SLBT.

Public API
----------
plot_html(model, output_file, title, visual_pruning, color_palette,
          gradient_colors)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

from .base import Node


DEFAULT_COLORS = [
    "#FF6384", "#36A2EB", "#FFCE56", "#4CAF50",
    "#9966FF", "#D81B60", "#00ACC1", "#8D6E63", "#FF9800",
]

# Default 2-stop gradient for twoClass (blue → red).
DEFAULT_GRADIENT = ["#3B82F6", "#EF4444"]


# ============================================================
#  Layout helpers
# ============================================================

def _compute_offsets_for_position(pos: int, depth: int) -> Tuple[float, float]:
    var  = pos
    l_c  = 0.0
    r_c  = 0.0
    incr = 1.0 / (2 ** depth)
    while var > 1:
        l_c  += (-2 * (var % 2) + 1) * incr
        r_c  += (2  * (var % 2) - 1) * incr
        incr *= 2.0
        var  //= 2
    return l_c, r_c


def compute_tree_layout(root: Node):
    """Return (max_depth, max_lc, max_rc, max_impurity_decrease, max_cumulative_path_tau)."""
    if root is None:
        return 0, 0.0, 0.0, 0.0, 0.0

    max_depth    = 0
    max_lc       = 0.0
    max_rc       = 0.0
    max_imp      = 0.0
    max_cum_tau  = 0.0

    stack = [(root, 0)]
    while stack:
        node, d = stack.pop()
        if node is None:
            continue
        if d > max_depth:
            max_depth = d
        if float(node.impurity_decrease) > max_imp:
            max_imp = node.impurity_decrease
        cum_tau = node.cumulative_path_tau if node.cumulative_path_tau is not None else 0.0
        if cum_tau > max_cum_tau:
            max_cum_tau = cum_tau
        lc_node, rc_node = _compute_offsets_for_position(node.position, d)
        if lc_node > max_lc: max_lc = lc_node
        if rc_node > max_rc: max_rc = rc_node
        if node.left  is not None: stack.append((node.left,  d + 1))
        if node.right is not None: stack.append((node.right, d + 1))

    return max_depth, max_lc, max_rc, max_imp, max_cum_tau


def _custom_round(value, decimals=2):
    if value is None:
        return None
    return round(float(value), decimals)


# ============================================================
#  Tree → dict  (two modes)
# ============================================================

def _recurse_vp(node: Node):
    """Serialise node for visual-pruning mode (y-axis = impurity_decrease)."""
    dist = ", ".join(
        f"{label}={_custom_round(prob, 2)}"
        for label, prob in zip(node.labels.tolist(), node.distribution.tolist())
    )
    suggested = 1 if node.suggested_pruning is not None else 0

    if node._is_leaf_node():
        gcr = None
        if node.GCR is not None:
            gcr = ", ".join(
                f"{label}={_custom_round(prob, 2)}"
                for label, prob in zip(node.labels.tolist(), node.GCR)
            )
        return {
            "isLeaf":                          1,
            "suggested":                       suggested,
            "distribution":                    dist,
            "distArray":                       node.distribution.tolist(),
            "position":                        node.position,
            "value":                           str(_custom_round(node.value, 2)) if isinstance(node.value, float) else str(node.value),
            "impurity":                        float(node.impurity),
            "labels":                          int(node.N),
            "impurity_decrement":              float(node.impurity_decrease),
            "tree_partial_impurity_reduction": float(node.tree_partial_impurity_reduction),
            "tau_decrease":                    float(node.tau_decrease)            if node.tau_decrease            is not None else 0.0,
            "cumulative_path_tau":             float(node.cumulative_path_tau)     if node.cumulative_path_tau     is not None else 0.0,
            "tree_partial_tau_reduction":      float(node.tree_partial_tau_reduction) if node.tree_partial_tau_reduction is not None else 0.0,
            "labArray":                        node.labels.tolist(),
            "gcr":                             gcr,
            "y_stats":                         node.y_stats,
        }

    treshold = ", ".join(str(x) for x in node.treshold)
    return {
        "isLeaf":                          0,
        "suggested":                       suggested,
        "feature":                         node.feature,
        "distribution":                    dist,
        "distArray":                       node.distribution.tolist(),
        "treshold":                        treshold,
        "position":                        node.position,
        "gpi":                             float(node.gpi),
        "pi":                              float(node.pi),
        "impurity":                        float(node.impurity),
        "impurity_decrement":              float(node.impurity_decrease),
        "tree_partial_impurity_reduction": float(node.tree_partial_impurity_reduction),
        "tau_decrease":                    float(node.tau_decrease)            if node.tau_decrease            is not None else 0.0,
        "cumulative_path_tau":             float(node.cumulative_path_tau)     if node.cumulative_path_tau     is not None else 0.0,
        "tree_partial_tau_reduction":      float(node.tree_partial_tau_reduction) if node.tree_partial_tau_reduction is not None else 0.0,
        "labels":                          int(node.N),
        "labArray":                        node.labels.tolist(),
        "y_stats":                         node.y_stats,
        "children": [
            _recurse_vp(node.left),
            _recurse_vp(node.right),
        ],
    }


def _recurse_std(node: Node, depth: int = 0):
    """Serialise node for standard mode (uniform y-step per level)."""
    dist = ", ".join(
        f"{label}={_custom_round(prob, 2)}"
        for label, prob in zip(node.labels.tolist(), node.distribution.tolist())
    )

    if node._is_leaf_node():
        gcr = None
        if node.GCR is not None:
            gcr = ", ".join(
                f"{label}={_custom_round(prob, 2)}"
                for label, prob in zip(node.labels.tolist(), node.GCR)
            )
        return {
            "isLeaf":       1,
            "distribution": dist,
            "distArray":    node.distribution.tolist(),
            "position":     node.position,
            "value":        str(_custom_round(node.value, 2)) if isinstance(node.value, float) else str(node.value),
            "impurity":     float(node.impurity),
            "labels":       int(node.N),
            "depth":        depth,
            "labArray":     node.labels.tolist(),
            "gcr":          gcr,
            "y_stats":      node.y_stats,
        }

    treshold = ", ".join(str(x) for x in node.treshold)
    return {
        "isLeaf":       0,
        "feature":      node.feature,
        "distribution": dist,
        "distArray":    node.distribution.tolist(),
        "treshold":     treshold,
        "position":     node.position,
        "gpi":          float(node.gpi),
        "pi":           float(node.pi),
        "impurity":     float(node.impurity),
        "labels":       int(node.N),
        "depth":        depth,
        "labArray":     node.labels.tolist(),
        "y_stats":      node.y_stats,
        "children": [
            _recurse_std(node.left,  depth + 1),
            _recurse_std(node.right, depth + 1),
        ],
    }


# ============================================================
#  Shared JavaScript block
# ============================================================

def _js_common():
    return """
        // ---- Legend / gradient bar ----
        if (plotData.is_twoclass) {
            const bar = document.createElement('div');
            bar.id = 'gradient-bar';

            const track = document.createElement('div');
            track.id = 'gradient-track';
            track.style.background = 'linear-gradient(to right, ' + plotData.gradient_colors.join(', ') + ')';

            const labelsDiv = document.createElement('div');
            labelsDiv.id = 'gradient-labels';
            const nStops = 5;
            const rangeY = plotData.y_global_max - plotData.y_global_min;
            for (let i = 0; i < nStops; i++) {
                const span = document.createElement('span');
                span.textContent = (plotData.y_global_min + (i / (nStops - 1)) * rangeY).toFixed(2);
                labelsDiv.appendChild(span);
            }
            bar.appendChild(track);
            bar.appendChild(labelsDiv);
            document.body.appendChild(bar);
        } else {
            const legendContainer = document.createElement('div');
            legendContainer.id = 'legend';
            for (let i = 0; i < plotData.labels.length; i++) {
                const item      = document.createElement('div');
                item.className  = 'legend-item';
                const colorBox  = document.createElement('div');
                colorBox.className = 'color-box';
                colorBox.style.backgroundColor = plotData.colors[i];
                const label     = document.createElement('span');
                label.textContent = plotData.labels[i];
                item.appendChild(colorBox);
                item.appendChild(label);
                legendContainer.appendChild(item);
            }
            document.body.appendChild(legendContainer);
        }

        // ---- Gradient interpolation (twoClass) ----
        function interpolateGradient(t, colors) {
            function hexToRgb(hex) {
                return [
                    parseInt(hex.slice(1, 3), 16),
                    parseInt(hex.slice(3, 5), 16),
                    parseInt(hex.slice(5, 7), 16)
                ];
            }
            function lerp(a, b, u) { return Math.round(a + (b - a) * u); }
            function rgbToHex(r, g, b) {
                return '#' + [r, g, b].map(x => x.toString(16).padStart(2, '0')).join('');
            }
            t = Math.max(0, Math.min(1, t));
            if (colors.length === 2) {
                const [c0, c1] = colors.map(hexToRgb);
                return rgbToHex(lerp(c0[0], c1[0], t), lerp(c0[1], c1[1], t), lerp(c0[2], c1[2], t));
            }
            // 3+ colors: piecewise linear across equal segments
            const n = colors.length - 1;
            const seg = Math.min(Math.floor(t * n), n - 1);
            const u   = (t * n) - seg;
            const [cA, cB] = [colors[seg], colors[seg + 1]].map(hexToRgb);
            return rgbToHex(lerp(cA[0], cB[0], u), lerp(cA[1], cB[1], u), lerp(cA[2], cB[2], u));
        }

        // ---- Pie chart (classification) ----
        function drawPieChart(canvas, distArray, labArray, isHighlighted) {
            let ctx = canvas.getContext("2d");
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            let total = distArray.reduce((sum, val) => sum + val, 0);
            let startAngle = 0;
            distArray.forEach((value, index) => {
                let sliceAngle = (value / total) * 2 * Math.PI;
                ctx.beginPath();
                ctx.moveTo(18, 18);
                ctx.arc(18, 18, 16, startAngle, startAngle + sliceAngle);
                ctx.closePath();
                plotData.labels.forEach((valueLab, indexLab) => {
                    if (valueLab == labArray[index]) {
                        ctx.fillStyle = plotData.colors[indexLab % plotData.colors.length];
                    }
                });
                ctx.fill();
                startAngle += sliceAngle;
            });
            ctx.beginPath();
            ctx.arc(18, 18, 17, 0, 2 * Math.PI);
            ctx.strokeStyle = isHighlighted ? "orange" : "blue";
            ctx.lineWidth = isHighlighted ? 3 : 2;
            ctx.stroke();
        }

        // ---- Boxplot (twoClass) ----
        // Redesigned: circular clip, fully opaque fill, white median, thick
        // colored ring, orange ring when highlighted.
        function drawBoxPlot(canvas, stats, isHighlighted) {
            const ctx = canvas.getContext("2d");
            const W = canvas.width, H = canvas.height;
            const cx = W / 2, cy = H / 2, r = H / 2 - 1;
            const PAD = 4, plotW = W - 2 * PAD;
            const yMin  = plotData.y_global_min;
            const yMax  = plotData.y_global_max;
            const range = (yMax - yMin) || 1;

            function scaleX(v) { return PAD + ((v - yMin) / range) * plotW; }

            const t     = (stats.median - yMin) / range;
            const color = interpolateGradient(t, plotData.gradient_colors);

            ctx.clearRect(0, 0, W, H);

            // --- clip all drawing to the circle ---
            ctx.save();
            ctx.beginPath();
            ctx.arc(cx, cy, r - 2, 0, 2 * Math.PI);
            ctx.clip();

            // Light background tint from gradient color
            ctx.fillStyle = color + "28";   // ~16% opacity tint
            ctx.fillRect(0, 0, W, H);

            // Whisker line
            ctx.beginPath();
            ctx.moveTo(scaleX(stats.wlo), cy);
            ctx.lineTo(scaleX(stats.whi), cy);
            ctx.strokeStyle = "#333"; ctx.lineWidth = 2; ctx.stroke();

            // Whisker caps
            ctx.beginPath();
            ctx.moveTo(scaleX(stats.wlo), cy - 5); ctx.lineTo(scaleX(stats.wlo), cy + 5);
            ctx.moveTo(scaleX(stats.whi), cy - 5); ctx.lineTo(scaleX(stats.whi), cy + 5);
            ctx.strokeStyle = "#333"; ctx.lineWidth = 2; ctx.stroke();

            // IQR box — fully opaque fill, dark border
            const boxX = scaleX(stats.q1);
            const boxW = Math.max(scaleX(stats.q3) - scaleX(stats.q1), 2);
            const boxH = 14;
            ctx.beginPath();
            ctx.rect(boxX, cy - boxH / 2, boxW, boxH);
            ctx.fillStyle = color;         // fully opaque
            ctx.fill();
            ctx.strokeStyle = "#222"; ctx.lineWidth = 1; ctx.stroke();

            // Median line — white for contrast
            ctx.beginPath();
            ctx.moveTo(scaleX(stats.median), cy - boxH / 2);
            ctx.lineTo(scaleX(stats.median), cy + boxH / 2);
            ctx.strokeStyle = "#fff"; ctx.lineWidth = 3; ctx.stroke();

            ctx.restore();  // end clip

            // --- outer ring: gradient color normally, orange when highlighted ---
            ctx.beginPath();
            ctx.arc(cx, cy, r, 0, 2 * Math.PI);
            ctx.strokeStyle = isHighlighted ? "orange" : color;
            ctx.lineWidth   = isHighlighted ? 4 : 3;
            ctx.stroke();
        }

        // ---- Dispatch: pie or boxplot (isHighlighted optional) ----
        function drawNodeChart(canvas, node, isHighlighted) {
            isHighlighted = !!isHighlighted;
            if (plotData.is_twoclass && node.y_stats) {
                drawBoxPlot(canvas, node.y_stats, isHighlighted);
            } else if (node.distArray && node.labArray) {
                drawPieChart(canvas, node.distArray, node.labArray, isHighlighted);
            }
        }

        function highlightImpurityLine(nodeId) {
            ['impurity-line','suggested-line'].forEach(cls => {
                document.querySelectorAll('.' + cls + '[data-node-id="' + nodeId + '"]').forEach(el => el.classList.add('highlighted'));
            });
            ['impurity-label-left','impurity-label-right','suggested-label-left','suggested-label-right'].forEach(cls => {
                document.querySelectorAll('.' + cls + '[data-node-id="' + nodeId + '"]').forEach(el => el.classList.add('highlighted'));
            });
        }

        function unhighlightImpurityLine(nodeId) {
            ['impurity-line','suggested-line'].forEach(cls => {
                document.querySelectorAll('.' + cls + '[data-node-id="' + nodeId + '"]').forEach(el => el.classList.remove('highlighted'));
            });
            ['impurity-label-left','impurity-label-right','suggested-label-left','suggested-label-right'].forEach(cls => {
                document.querySelectorAll('.' + cls + '[data-node-id="' + nodeId + '"]').forEach(el => el.classList.remove('highlighted'));
            });
        }

        function highlightSubtree(currentImpurityDecrease) {
            const allNodes = document.querySelectorAll('[data-node-id]');
            const toHL = new Set();
            allNodes.forEach(el => {
                if (parseFloat(el.getAttribute('data-impurity-decrease')) < currentImpurityDecrease)
                    toHL.add(parseInt(el.getAttribute('data-node-id')));
            });
            const copy = new Set(toHL);
            copy.forEach(id => { toHL.add(2 * id); toHL.add(2 * id + 1); });
            toHL.forEach(nodeId => {
                document.querySelectorAll('.node[data-node-id="' + nodeId + '"], .leaf[data-node-id="' + nodeId + '"]').forEach(el => {
                    el.classList.add('highlighted-node');
                    const canvas = el.querySelector('canvas');
                    // Redraw with highlighted=true to use orange ring correctly
                    if (canvas && canvas._lbNodeData) drawNodeChart(canvas, canvas._lbNodeData, true);
                });
                document.querySelectorAll('.square[data-node-id="' + nodeId + '"]').forEach(el => el.classList.add('highlighted-branch'));
            });
        }

        function unhighlightSubtree() {
            document.querySelectorAll('.highlighted-node').forEach(el => {
                el.classList.remove('highlighted-node');
                const canvas = el.querySelector('canvas');
                // Restore original chart (highlighted=false restores gradient/blue ring)
                if (canvas && canvas._lbNodeData) drawNodeChart(canvas, canvas._lbNodeData, false);
            });
            document.querySelectorAll('.highlighted-branch').forEach(el => el.classList.remove('highlighted-branch'));
        }

        function showTooltip(event, node, top, left) {
            const tooltip = document.getElementById('tooltip');
            let text;
            if (plotData.is_twoclass && node.y_stats) {
                const s = node.y_stats;
                if (node.isLeaf == 1) {
                    text = "Id: " + node.position
                         + "\\nN: " + node.labels
                         + "\\nMediana: " + s.median.toFixed(3)
                         + "\\nQ1 - Q3: [" + s.q1.toFixed(3) + ", " + s.q3.toFixed(3) + "]"
                         + "\\nMedia: " + s.mean.toFixed(3)
                         + "\\nPrediction: " + node.value;
                } else {
                    text = "Id: " + node.position
                         + "\\nN: " + node.labels
                         + "\\nMediana: " + s.median.toFixed(3)
                         + "\\nQ1 - Q3: [" + s.q1.toFixed(3) + ", " + s.q3.toFixed(3) + "]"
                         + "\\nFeature: " + node.feature
                         + "\\nGPI: " + node.gpi.toFixed(3)
                         + "\\nPPI: " + node.pi.toFixed(3)
                         + "\\nVar. riduzione: " + node.impurity.toFixed(4);
                }
            } else {
                if (node.isLeaf == 1) {
                    text = "Id: " + node.position + "\\nN: " + node.labels + "\\nDistribution: [" + node.distribution + "]\\nGCR: [" + node.gcr + "]\\nImpurity: " + node.impurity.toFixed(3) + "\\nPrediction: " + node.value;
                } else {
                    text = "Id: " + node.position + "\\nN: " + node.labels + "\\nDistribution: [" + node.distribution + "]\\nFeature: " + node.feature + "\\nThreshold left: [" + node.treshold + "]\\nGPI: " + node.gpi.toFixed(3) + "\\nPPI: " + node.pi.toFixed(3) + "\\nImpurity: " + node.impurity.toFixed(3);
                }
            }
            tooltip.innerText = text;
            if (left > 50) {
                tooltip.style.left = left + "%"; tooltip.style.top = top + "%"; tooltip.classList.add("t_r");
            } else {
                tooltip.style.left = left + "%"; tooltip.style.top = top + "%"; tooltip.classList.add("t_l");
            }
            tooltip.style.visibility = 'visible'; tooltip.style.opacity = 1;
        }

        function hideTooltip() {
            const tooltip = document.getElementById('tooltip');
            tooltip.classList.remove("t_r", "t_l");
            tooltip.style.visibility = 'hidden'; tooltip.style.opacity = 0;
        }

        const tree = document.getElementById("tree");
        let d;
    """


# ============================================================
#  HTML templates
# ============================================================

_COMMON_CSS = """
    body { font-family: Arial, sans-serif; text-align: center }
    .tree-container { display: flex; justify-content: center; margin-top: 20px }
    .node { border: 2px solid blue; border-radius: 100%; background-color: lightcyan; color: lightcyan; display: inline-block; position: absolute; width: 24px; height: 24px }
    .square { position: absolute; transform: translate(14px, 14px); z-index: -1 }
    .leaf { display: inline-block; font-weight: bolder; position: absolute; line-height: 24px }
    .d_r:after { content: ""; width: 100%; height: 100%; position: absolute; top: 0; left: 0; background: linear-gradient(to top left, transparent calc(50% - 1px), blue, transparent calc(50% + 1px)) }
    .d_l:after { content: ""; width: 100%; height: 100%; position: absolute; top: 0; left: 0; background: linear-gradient(to top right, transparent calc(50% - 1px), blue, transparent calc(50% + 1px)) }
    .tooltip { position: absolute; z-index: 1; background-color: rgba(0,0,0,0.8); color: white; padding: 8px; border-radius: 5px; white-space: nowrap; visibility: hidden; opacity: 0; transition: opacity 0.3s; font-size: 14px; width: fit-content }
    .t_l { transform: translate(35px, -5px) }
    .t_r { transform: translateX(-100%) translate(-8px, -5px) }
    .leaf_value { display: inline-block; font-weight: bolder; position: absolute; line-height: 24px; transform: translate(6px, 30px) }
    #legend { display: flex; flex-wrap: wrap; gap: 8px; position: absolute; bottom: 3%; width: 100%; justify-content: center; font-family: sans-serif }
    .legend-item { display: flex; align-items: center; gap: 6px; margin-right: 12px }
    .color-box { width: 16px; height: 16px; border: 1px solid #000; box-sizing: border-box }
    #gradient-bar { position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%); width: 280px; font-family: sans-serif; z-index: 10; }
    #gradient-track { height: 14px; border-radius: 6px; border: 1px solid #bbb; margin-bottom: 5px; }
    #gradient-labels { display: flex; justify-content: space-between; font-size: 11px; color: #444; }
"""


def _html_visual_pruning(tree_JSON, plot_JSON, title):
    extra_css = """
    .impurity-line { position: absolute; width: 90%; border-top: 1px dashed #cccccc; z-index: -2; transition: border-color 0.3s; transform: translateY(12px) }
    .suggested-line { position: absolute; width: 90%; border-top: 1px dashed #e61414; z-index: -2; transition: border-color 0.3s; transform: translateY(12px) }
    .impurity-line.highlighted { border-top: 2px dashed #4a90e2; z-index: -1 }
    .suggested-line.highlighted { border-top: 2px dashed #e61414; z-index: -1 }
    .impurity-label-left  { position: absolute; font-size: 11px; color: #666; background-color: rgba(255,255,255,0.8); padding: 2px 5px; border-radius: 3px; transition: color 0.3s; z-index: 0; right: 0%; transform: translateY(12px) }
    .impurity-label-right { position: absolute; font-size: 11px; color: #666; background-color: rgba(255,255,255,0.8); padding: 2px 5px; border-radius: 3px; transition: color 0.3s; z-index: 0; left:  0%; transform: translateY(12px) }
    .suggested-label-left  { position: absolute; font-size: 11px; color: #e61414; background-color: rgba(255,255,255,0.8); padding: 2px 5px; border-radius: 3px; z-index: 1; right: 0%; transform: translateY(12px) }
    .suggested-label-right { position: absolute; font-size: 11px; color: #e61414; background-color: rgba(255,255,255,0.8); padding: 2px 5px; border-radius: 3px; z-index: 1; left:  0%; transform: translateY(12px) }
    .impurity-label-left.highlighted  { color: #ffffff; background-color: rgba(74,144,226,0.95); font-weight: bold; z-index: 11 }
    .impurity-label-right.highlighted { color: #ffffff; background-color: rgba(74,144,226,0.95); font-weight: bold; z-index: 11 }
    .suggested-label-left.highlighted  { color: #ffffff; background-color: #e61414; font-weight: bold; z-index: 11 }
    .suggested-label-right.highlighted { color: #ffffff; background-color: #e61414; font-weight: bold; z-index: 11 }
    .square.highlighted-branch:after { background: linear-gradient(to top left, transparent calc(50% - 1px), orange, transparent calc(50% + 1px)) !important }
    .d_l.highlighted-branch:after    { background: linear-gradient(to top right, transparent calc(50% - 1px), orange, transparent calc(50% + 1px)) !important }
    """
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>{_COMMON_CSS}{extra_css}</style>
</head>
<body>
    <h1>{title}</h1>
    <div class="tree-container" id="tree"></div>
    <div id="tooltip" class="tooltip"></div>
    <script>
        const treeData = {tree_JSON};
        const plotData = {plot_JSON};

        function iter(node, left, d, h, prev_impurity) {{
            let impurityLineElement = document.createElement("div");
            impurityLineElement.style.top = (15 + node.impurity_decrement * h) + "%";
            impurityLineElement.setAttribute("data-node-id", node.position);

            let impurityLabelLeft = document.createElement("div");
            impurityLabelLeft.style.top = (15 + node.impurity_decrement * h - 1) + "%";
            impurityLabelLeft.innerText = "V_t(T): " + node.impurity_decrement.toFixed(3);
            impurityLabelLeft.setAttribute("data-node-id", node.position);

            let impurityLabelRight = document.createElement("div");
            impurityLabelRight.style.top = (15 + node.impurity_decrement * h - 1) + "%";
            impurityLabelRight.innerText = "V(T): " + node.tree_partial_impurity_reduction.toFixed(3);
            impurityLabelRight.setAttribute("data-node-id", node.position);

            if (node.suggested == 1) {{
                impurityLineElement.classList.add("suggested-line");
                impurityLabelLeft.classList.add("suggested-label-left");
                impurityLabelRight.classList.add("suggested-label-right");
            }} else {{
                impurityLineElement.classList.add("impurity-line");
                impurityLabelLeft.classList.add("impurity-label-left");
                impurityLabelRight.classList.add("impurity-label-right");
            }}
            tree.appendChild(impurityLabelLeft);
            tree.appendChild(impurityLabelRight);
            tree.appendChild(impurityLineElement);

            if (node.impurity_decrement != 0.0) {{
                if (node.position % 2 === 0) {{
                    let el = document.createElement("div");
                    el.classList.add("square", "d_r");
                    el.style.left   = left + "%";
                    el.style.top    = 15 + prev_impurity * h + "%";
                    el.style.height = (node.impurity_decrement - prev_impurity) * h + "%";
                    el.style.width  = d + "%";
                    el.setAttribute("data-node-id", node.position);
                    el.setAttribute("data-impurity-decrease", node.impurity_decrement);
                    tree.appendChild(el);
                }} else {{
                    let el = document.createElement("div");
                    el.classList.add("square", "d_l");
                    el.style.right  = (100 - left) + "%";
                    el.style.top    = 15 + prev_impurity * h + "%";
                    el.style.height = (node.impurity_decrement - prev_impurity) * h + "%";
                    el.style.width  = d + "%";
                    el.setAttribute("data-node-id", node.position);
                    el.setAttribute("data-impurity-decrease", node.impurity_decrement);
                    tree.appendChild(el);
                }}
            }}

            if (node.isLeaf == 1) {{
                let nodeElement = document.createElement("div");
                nodeElement.classList.add("leaf");
                nodeElement.style.left = left + "%";
                nodeElement.style.top  = 15 + node.impurity_decrement * h + "%";
                nodeElement.setAttribute("data-node-id", node.position);
                nodeElement.setAttribute("data-impurity-decrease", node.impurity_decrement);
                nodeElement.onmouseover = function(event) {{ showTooltip(event, node, 15 + node.impurity_decrement * h, left); highlightImpurityLine(node.position); highlightSubtree(node.impurity_decrement); }};
                nodeElement.onmouseout  = function() {{ hideTooltip(); unhighlightImpurityLine(node.position); unhighlightSubtree(); }};
                let nodeValue = document.createElement("div");
                nodeValue.classList.add("leaf_value");
                nodeValue.style.left = left + "%";
                nodeValue.style.top  = 15 + node.impurity_decrement * h + "%";
                nodeValue.innerText  = node.value;
                let canvas = document.createElement("canvas");
                canvas.width = 36; canvas.height = 36;
                canvas.style.position = "absolute"; canvas.style.top = "-6px"; canvas.style.left = "-6px";
                canvas._lbNodeData = node;
                nodeElement.appendChild(canvas);
                tree.appendChild(nodeElement);
                tree.appendChild(nodeValue);
                drawNodeChart(canvas, node);
                return;
            }}

            let nodeElement = document.createElement("div");
            nodeElement.classList.add("node");
            nodeElement.style.left = left + "%";
            nodeElement.style.top  = 15 + node.impurity_decrement * h + "%";
            nodeElement.setAttribute("data-node-id", node.position);
            nodeElement.setAttribute("data-impurity-decrease", node.impurity_decrement);
            nodeElement.onmouseover = function(event) {{ showTooltip(event, node, 15 + node.impurity_decrement * h, left); highlightImpurityLine(node.position); highlightSubtree(node.impurity_decrement); }};
            nodeElement.onmouseout  = function() {{ hideTooltip(); unhighlightImpurityLine(node.position); unhighlightSubtree(); }};
            let canvas = document.createElement("canvas");
            canvas.width = 36; canvas.height = 36;
            canvas.style.position = "absolute"; canvas.style.top = "-6px"; canvas.style.left = "-6px";
            canvas._lbNodeData = node;
            nodeElement.appendChild(canvas);
            tree.appendChild(nodeElement);
            drawNodeChart(canvas, node);

            iter(node.children[0], left - d / 2, d / 2, h, node.impurity_decrement);
            iter(node.children[1], left + d / 2, d / 2, h, node.impurity_decrement);
        }}

        {_js_common()}

        d = 80 / (plotData.l_c + plotData.r_c);
        h = 65 / plotData.max_imp_decrease;
        iter(treeData, 10 + d * plotData.l_c, d, h, 0);
    </script>
</body>
</html>"""


def _html_visual_pruning_tau(tree_JSON, plot_JSON, title):
    extra_css = """
    .impurity-line { position: absolute; width: 90%; border-top: 1px dashed #cccccc; z-index: -2; transition: border-color 0.3s; transform: translateY(12px) }
    .suggested-line { position: absolute; width: 90%; border-top: 1px dashed #e61414; z-index: -2; transition: border-color 0.3s; transform: translateY(12px) }
    .impurity-line.highlighted { border-top: 2px dashed #4a90e2; z-index: -1 }
    .suggested-line.highlighted { border-top: 2px dashed #e61414; z-index: -1 }
    .impurity-label-left  { position: absolute; font-size: 11px; color: #666; background-color: rgba(255,255,255,0.8); padding: 2px 5px; border-radius: 3px; transition: color 0.3s; z-index: 0; right: 0%; transform: translateY(12px) }
    .impurity-label-right { position: absolute; font-size: 11px; color: #666; background-color: rgba(255,255,255,0.8); padding: 2px 5px; border-radius: 3px; transition: color 0.3s; z-index: 0; left:  0%; transform: translateY(12px) }
    .suggested-label-left  { position: absolute; font-size: 11px; color: #e61414; background-color: rgba(255,255,255,0.8); padding: 2px 5px; border-radius: 3px; z-index: 1; right: 0%; transform: translateY(12px) }
    .suggested-label-right { position: absolute; font-size: 11px; color: #e61414; background-color: rgba(255,255,255,0.8); padding: 2px 5px; border-radius: 3px; z-index: 1; left:  0%; transform: translateY(12px) }
    .impurity-label-left.highlighted  { color: #ffffff; background-color: rgba(74,144,226,0.95); font-weight: bold; z-index: 11 }
    .impurity-label-right.highlighted { color: #ffffff; background-color: rgba(74,144,226,0.95); font-weight: bold; z-index: 11 }
    .suggested-label-left.highlighted  { color: #ffffff; background-color: #e61414; font-weight: bold; z-index: 11 }
    .suggested-label-right.highlighted { color: #ffffff; background-color: #e61414; font-weight: bold; z-index: 11 }
    .square.highlighted-branch:after {{ background: linear-gradient(to top left, transparent calc(50% - 1px), orange, transparent calc(50% + 1px)) !important }}
    .d_l.highlighted-branch:after    {{ background: linear-gradient(to top right, transparent calc(50% - 1px), orange, transparent calc(50% + 1px)) !important }}
    """
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>{_COMMON_CSS}{extra_css}</style>
</head>
<body>
    <h1>{title}</h1>
    <div class="tree-container" id="tree"></div>
    <div id="tooltip" class="tooltip"></div>
    <script>
        const treeData = {tree_JSON};
        const plotData = {plot_JSON};

        function iter(node, left, d, h, prev_tau) {{
            let impurityLineElement = document.createElement("div");
            impurityLineElement.style.top = (15 + node.cumulative_path_tau * h) + "%";
            impurityLineElement.setAttribute("data-node-id", node.position);

            let impurityLabelLeft = document.createElement("div");
            impurityLabelLeft.style.top = (15 + node.cumulative_path_tau * h - 1) + "%";
            impurityLabelLeft.innerText = "τ_t(T): " + node.cumulative_path_tau.toFixed(3);
            impurityLabelLeft.setAttribute("data-node-id", node.position);

            let impurityLabelRight = document.createElement("div");
            impurityLabelRight.style.top = (15 + node.cumulative_path_tau * h - 1) + "%";
            impurityLabelRight.innerText = "τ(T): " + node.tree_partial_tau_reduction.toFixed(3);
            impurityLabelRight.setAttribute("data-node-id", node.position);

            if (node.suggested == 1) {{
                impurityLineElement.classList.add("suggested-line");
                impurityLabelLeft.classList.add("suggested-label-left");
                impurityLabelRight.classList.add("suggested-label-right");
            }} else {{
                impurityLineElement.classList.add("impurity-line");
                impurityLabelLeft.classList.add("impurity-label-left");
                impurityLabelRight.classList.add("impurity-label-right");
            }}
            tree.appendChild(impurityLabelLeft);
            tree.appendChild(impurityLabelRight);
            tree.appendChild(impurityLineElement);

            if (node.cumulative_path_tau != 0.0) {{
                if (node.position % 2 === 0) {{
                    let el = document.createElement("div");
                    el.classList.add("square", "d_r");
                    el.style.left   = left + "%";
                    el.style.top    = 15 + prev_tau * h + "%";
                    el.style.height = (node.cumulative_path_tau - prev_tau) * h + "%";
                    el.style.width  = d + "%";
                    el.setAttribute("data-node-id", node.position);
                    el.setAttribute("data-impurity-decrease", node.cumulative_path_tau);
                    tree.appendChild(el);
                }} else {{
                    let el = document.createElement("div");
                    el.classList.add("square", "d_l");
                    el.style.right  = (100 - left) + "%";
                    el.style.top    = 15 + prev_tau * h + "%";
                    el.style.height = (node.cumulative_path_tau - prev_tau) * h + "%";
                    el.style.width  = d + "%";
                    el.setAttribute("data-node-id", node.position);
                    el.setAttribute("data-impurity-decrease", node.cumulative_path_tau);
                    tree.appendChild(el);
                }}
            }}

            if (node.isLeaf == 1) {{
                let nodeElement = document.createElement("div");
                nodeElement.classList.add("leaf");
                nodeElement.style.left = left + "%";
                nodeElement.style.top  = 15 + node.cumulative_path_tau * h + "%";
                nodeElement.setAttribute("data-node-id", node.position);
                nodeElement.setAttribute("data-impurity-decrease", node.cumulative_path_tau);
                nodeElement.onmouseover = function(event) {{ showTooltip(event, node, 15 + node.cumulative_path_tau * h, left); highlightImpurityLine(node.position); highlightSubtree(node.cumulative_path_tau); }};
                nodeElement.onmouseout  = function() {{ hideTooltip(); unhighlightImpurityLine(node.position); unhighlightSubtree(); }};
                let nodeValue = document.createElement("div");
                nodeValue.classList.add("leaf_value");
                nodeValue.style.left = left + "%";
                nodeValue.style.top  = 15 + node.cumulative_path_tau * h + "%";
                nodeValue.innerText  = node.value;
                let canvas = document.createElement("canvas");
                canvas.width = 36; canvas.height = 36;
                canvas.style.position = "absolute"; canvas.style.top = "-6px"; canvas.style.left = "-6px";
                canvas._lbNodeData = node;
                nodeElement.appendChild(canvas);
                tree.appendChild(nodeElement);
                tree.appendChild(nodeValue);
                drawNodeChart(canvas, node);
                return;
            }}

            let nodeElement = document.createElement("div");
            nodeElement.classList.add("node");
            nodeElement.style.left = left + "%";
            nodeElement.style.top  = 15 + node.cumulative_path_tau * h + "%";
            nodeElement.setAttribute("data-node-id", node.position);
            nodeElement.setAttribute("data-impurity-decrease", node.cumulative_path_tau);
            nodeElement.onmouseover = function(event) {{ showTooltip(event, node, 15 + node.cumulative_path_tau * h, left); highlightImpurityLine(node.position); highlightSubtree(node.cumulative_path_tau); }};
            nodeElement.onmouseout  = function() {{ hideTooltip(); unhighlightImpurityLine(node.position); unhighlightSubtree(); }};
            let canvas = document.createElement("canvas");
            canvas.width = 36; canvas.height = 36;
            canvas.style.position = "absolute"; canvas.style.top = "-6px"; canvas.style.left = "-6px";
            canvas._lbNodeData = node;
            nodeElement.appendChild(canvas);
            tree.appendChild(nodeElement);
            drawNodeChart(canvas, node);

            iter(node.children[0], left - d / 2, d / 2, h, node.cumulative_path_tau);
            iter(node.children[1], left + d / 2, d / 2, h, node.cumulative_path_tau);
        }}

        {_js_common()}

        d = 80 / (plotData.l_c + plotData.r_c);
        h = 65 / (plotData.max_cum_tau || 1);
        iter(treeData, 10 + d * plotData.l_c, d, h, 0);
    </script>
</body>
</html>"""


def _html_standard(tree_JSON, plot_JSON, title):
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>{_COMMON_CSS}</style>
</head>
<body>
    <h1>{title}</h1>
    <div class="tree-container" id="tree"></div>
    <div id="tooltip" class="tooltip"></div>
    <script>
        const treeData = {tree_JSON};
        const plotData = {plot_JSON};

        function maxDepth(node) {{
            if (node.isLeaf == 1) return node.depth;
            return Math.max(maxDepth(node.children[0]), maxDepth(node.children[1]));
        }}

        function iter(node, left, d, h) {{
            let top = 15 + node.depth * h;
            if (node.depth > 0) {{
                if (node.position % 2 === 0) {{
                    let el = document.createElement("div");
                    el.classList.add("square", "d_r");
                    el.style.left   = left + "%";
                    el.style.top    = (top - h) + "%";
                    el.style.height = h + "%";
                    el.style.width  = d + "%";
                    tree.appendChild(el);
                }} else {{
                    let el = document.createElement("div");
                    el.classList.add("square", "d_l");
                    el.style.right  = (100 - left) + "%";
                    el.style.top    = (top - h) + "%";
                    el.style.height = h + "%";
                    el.style.width  = d + "%";
                    tree.appendChild(el);
                }}
            }}

            if (node.isLeaf == 1) {{
                let nodeElement = document.createElement("div");
                nodeElement.classList.add("leaf");
                nodeElement.style.left = left + "%";
                nodeElement.style.top  = top + "%";
                nodeElement.onmouseover = function(event) {{ showTooltip(event, node, top, left); }};
                nodeElement.onmouseout  = function() {{ hideTooltip(); }};
                let nodeValue = document.createElement("div");
                nodeValue.classList.add("leaf_value");
                nodeValue.style.left = left + "%";
                nodeValue.style.top  = top + "%";
                nodeValue.innerText  = node.value;
                let canvas = document.createElement("canvas");
                canvas.width = 36; canvas.height = 36;
                canvas.style.position = "absolute"; canvas.style.top = "-6px"; canvas.style.left = "-6px";
                canvas._lbNodeData = node;
                nodeElement.appendChild(canvas);
                tree.appendChild(nodeElement);
                tree.appendChild(nodeValue);
                drawNodeChart(canvas, node);
                return;
            }}

            let nodeElement = document.createElement("div");
            nodeElement.classList.add("node");
            nodeElement.style.left = left + "%";
            nodeElement.style.top  = top + "%";
            nodeElement.onmouseover = function(event) {{ showTooltip(event, node, top, left); }};
            nodeElement.onmouseout  = function() {{ hideTooltip(); }};
            let canvas = document.createElement("canvas");
            canvas.width = 36; canvas.height = 36;
            canvas.style.position = "absolute"; canvas.style.top = "-6px"; canvas.style.left = "-6px";
            canvas._lbNodeData = node;
            nodeElement.appendChild(canvas);
            tree.appendChild(nodeElement);
            drawNodeChart(canvas, node);

            iter(node.children[0], left - d / 2, d / 2, h);
            iter(node.children[1], left + d / 2, d / 2, h);
        }}

        {_js_common()}

        const md = maxDepth(treeData);
        const h  = md > 0 ? 65 / md : 65;
        d = 80 / (plotData.l_c + plotData.r_c);
        iter(treeData, 10 + d * plotData.l_c, d, h);
    </script>
</body>
</html>"""


# ============================================================
#  Public API
# ============================================================

def plot_html(
    model,
    output_file: str     = "tree.html",
    title: str           = "Tree Visualisation",
    visual_pruning: bool = False,
    vp_metric: str       = "impurity",
    color_palette        = None,
    gradient_colors      = None,
) -> None:
    """
    Generate an interactive HTML file visualising a fitted tree.

    Parameters
    ----------
    model : SCTree | SLBT
        Fitted model (must have a non-None ``.root`` attribute).
    output_file : str, default "tree.html"
        Path of the HTML file to create.
    title : str, default "Tree Visualisation"
        Page title.
    visual_pruning : bool, default False
        False → standard layout: uniform branch lengths per depth level.
        True  → visual-pruning layout: node vertical position proportional
                to ``vp_metric``; dashed horizontal lines show metric values.
    vp_metric : {"impurity", "tau"}, default "impurity"
        Metric for the visual-pruning y-axis.
        "impurity" → ``impurity_decrease`` (classic VP behaviour).
        "tau"      → ``cumulative_path_tau`` (Goodman-Kruskal τ VP).
        Ignored when ``visual_pruning=False``.
    color_palette : list[str], optional
        Hex colour list for the target classes (classification models).
        Defaults to DEFAULT_COLORS.  Ignored for twoClass.
    gradient_colors : list[str], optional
        List of 2 or 3 hex colours defining the gradient scale used for the
        twoClass boxplot visualisation.  The first colour corresponds to the
        minimum y value, the last to the maximum.  A middle colour (3-stop)
        creates a diverging scale.
        Defaults to ["#3B82F6", "#EF4444"] (blue → red).
        Ignored for non-twoClass models.

    Examples
    --------
    # Standard layout
    plot_html(clf, "tree.html")

    # Visual pruning — impurity (classic)
    plot_html(clf, "tree_vp.html", visual_pruning=True)

    # Visual pruning — tau
    plot_html(clf, "tree_tau.html", visual_pruning=True, vp_metric="tau")
    """
    if model.root is None:
        raise ValueError("Tree is empty. Call fit() before plot_html().")

    if vp_metric not in ("impurity", "tau"):
        raise ValueError("vp_metric must be 'impurity' or 'tau'.")

    is_twoclass = getattr(model, "model", None) == "twoClass"

    if color_palette is None:
        color_palette = DEFAULT_COLORS
    if gradient_colors is None:
        gradient_colors = DEFAULT_GRADIENT

    if len(gradient_colors) < 2 or len(gradient_colors) > 3:
        raise ValueError("gradient_colors must contain 2 or 3 hex colour strings.")

    _, l_c, r_c, max_imp, max_cum_tau = compute_tree_layout(model.root)

    # For twoClass: extract global y range from the root node y_stats
    y_global_min = 0.0
    y_global_max = 1.0
    if is_twoclass and model.root.y_stats is not None:
        y_global_min = model.root.y_stats["ymin"]
        y_global_max = model.root.y_stats["ymax"]

    plot_data = {
        "l_c":              l_c,
        "r_c":              r_c,
        "max_imp_decrease": max_imp,
        "max_cum_tau":      max_cum_tau,
        "colors":           color_palette,
        "labels":           model.root.labels.tolist(),
        "is_twoclass":      is_twoclass,
        "gradient_colors":  gradient_colors,
        "y_global_min":     y_global_min,
        "y_global_max":     y_global_max,
    }
    plot_JSON = json.dumps(plot_data, indent=2)

    if visual_pruning:
        tree_JSON = json.dumps(_recurse_vp(model.root), indent=2)
        if vp_metric == "tau":
            html = _html_visual_pruning_tau(tree_JSON, plot_JSON, title)
        else:
            html = _html_visual_pruning(tree_JSON, plot_JSON, title)
    else:
        tree_JSON = json.dumps(_recurse_std(model.root), indent=2)
        html      = _html_standard(tree_JSON, plot_JSON, title)

    Path(output_file).write_text(html, encoding="utf-8")
    print(f"Plot saved to: {output_file}")
