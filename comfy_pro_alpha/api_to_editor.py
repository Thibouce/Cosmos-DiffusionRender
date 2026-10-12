#!/usr/bin/env python3
"""Convertit un workflow API (build_workflow.py) en workflow format éditeur, node par node, pour l'ouvrir
et le modifier dans ComfyUI. Il lit les définitions de nodes de VOTRE ComfyUI (ordre des widgets, sorties).

  python api_to_editor.py workflow_api.json -o workflow_editor_full.json --url http://127.0.0.1:8188
  # ou, sans serveur joignable: ouvrir http://127.0.0.1:8188/object_info, enregistrer, puis
  python api_to_editor.py workflow_api.json --object-info object_info.json
"""
import argparse
import json
import urllib.request

PRIMS = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}
COL_W, GAP_X, GAP_Y = 380, 80, 30


def spec(info, name):
    for sect in ("required", "optional"):
        if name in info.get("input", {}).get(sect, {}):
            return info["input"][sect][name]
    return None


def input_order(info):
    return [(n, v) for sect in ("required", "optional") for n, v in info.get("input", {}).get(sect, {}).items()]


def kind(v):
    """-> (type, options, config, is_widget)"""
    t, cfg = v[0], (v[1] if len(v) > 1 and isinstance(v[1], dict) else {})
    if isinstance(t, list):
        return "COMBO", t, cfg, True
    if t == "COMBO":
        return "COMBO", cfg.get("options", []), cfg, True
    return t, None, cfg, t in PRIMS


def default(t, opts, cfg):
    if "default" in cfg:
        return cfg["default"]
    return {"COMBO": (opts or [""])[0] if opts else "", "BOOLEAN": False, "STRING": "", "INT": 0, "FLOAT": 0.0}.get(t)


def convert(api, oi):
    missing = sorted({n["class_type"] for n in api.values() if n["class_type"] not in oi})
    if missing:
        raise SystemExit("Nodes absents de ce ComfyUI (installer les packs puis relancer): " + ", ".join(missing))
    # profondeur topologique
    depth = {}

    def d(nid):
        if nid not in depth:
            ups = [v[0] for v in api[nid]["inputs"].values() if isinstance(v, list) and len(v) == 2 and v[0] in api]
            depth[nid] = 1 + max((d(u) for u in ups), default=-1)
        return depth[nid]

    for nid in api:
        d(nid)
    order = sorted(api, key=lambda i: (depth[i], int(i)))
    colY = {}
    nodes, links, out_links, link_id = {}, [], {}, 0
    for idx, nid in enumerate(order):
        n = api[nid]
        info = oi[n["class_type"]]
        ins, widgets = [], []
        for name, v in input_order(info):
            t, opts, cfg, is_w = kind(v)
            val = n["inputs"].get(name)
            linked = isinstance(val, list) and len(val) == 2 and str(val[0]) in api
            slot = {"name": name, "type": t if not is_w else t, "link": None}
            if is_w:
                slot["widget"] = {"name": name}
                widgets.append(default(t, opts, cfg) if (linked or name not in n["inputs"]) else val)
                if cfg.get("control_after_generate"):
                    widgets.append("fixed")
                if cfg.get("image_upload") or cfg.get("video_upload"):
                    widgets.append("image")
            ins.append(slot)
            if linked:
                link_id += 1
                src = str(val[0])
                st = info_out_type(oi[api[src]["class_type"]], val[1])
                slot["type"] = st if is_w else slot["type"]
                links.append([link_id, int(src), val[1], int(nid), len(ins) - 1, st])
                slot["link"] = link_id
                out_links.setdefault((src, val[1]), []).append(link_id)
        outs = [{"name": nm, "type": ty, "links": None, "slot_index": i}
                for i, (nm, ty) in enumerate(zip(info.get("output_name", info.get("output", [])), info.get("output", [])))]
        rows = max(len(ins), len(outs)) + (len([1 for s in ins if "widget" in s]))
        h = 40 + 24 * rows
        x = depth[nid] * (COL_W + GAP_X)
        y = colY.get(depth[nid], 0)
        colY[depth[nid]] = y + h + GAP_Y
        node = {"id": int(nid), "type": n["class_type"], "pos": [x, y], "size": [COL_W, h], "flags": {}, "order": idx,
                "mode": 0, "inputs": ins, "outputs": outs,
                "properties": {"Node name for S&R": n["class_type"]}, "widgets_values": widgets}
        title = n.get("_meta", {}).get("title")
        if title:
            node["title"] = title
        nodes[nid] = node
    for (src, slot), ids in out_links.items():
        nodes[src]["outputs"][slot]["links"] = ids
    return {"last_node_id": max(int(i) for i in api), "last_link_id": link_id,
            "nodes": [nodes[i] for i in order], "links": links, "groups": [], "config": {}, "extra": {}, "version": 0.4}


def info_out_type(info, slot):
    out = info.get("output", [])
    return out[slot] if slot < len(out) else "*"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("api_json")
    ap.add_argument("-o", "--output", default="workflow_editor_full.json")
    ap.add_argument("--url", default="http://127.0.0.1:8188")
    ap.add_argument("--object-info", help="fichier enregistré depuis /object_info (au lieu de --url)")
    a = ap.parse_args()
    api = json.load(open(a.api_json))
    oi = json.load(open(a.object_info)) if a.object_info else json.load(urllib.request.urlopen(a.url.rstrip("/") + "/object_info"))
    json.dump(convert(api, oi), open(a.output, "w"), indent=1)
    print(f"{len(api)} nodes -> {a.output}")


if __name__ == "__main__":
    main()
