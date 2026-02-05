### MindMap Data Schema Documentation

This document outlines the JSON structure required to populate the **Interactive Logic Map**. The system expects a single JSON object containing two main arrays: `nodes` and `edges`.

---

### 1. File Structure Overview

The JSON file must follow this root structure:

```json
{
  "nodes": [ ... ],
  "edges": [ ... ]
}

```

---

### 2. Nodes Configuration

Each object in the `nodes` array represents a functional block (Pump, Sensor, PID, etc.) on the canvas.

| Field      | Type     | Description                                                               |
| ---------- | -------- | ------------------------------------------------------------------------- |
| `id`       | `string` | Unique identifier (e.g., "P-101"). Must match `source`/`target` in edges. |
| `type`     | `string` | **Must be `"custom"**` to render the interactive faceplate.               |
| `position` | `object` | Coordinates on the canvas: `{ "x": 0, "y": 0 }`.                          |
| `data`     | `object` | The core configuration object. See **Data Object Details** below.         |

#### 2.1 The `data` Object

This object drives the UI and interactivity of the node.

- **`type`**: Determines the color and icon.
- `"EQUIPMENT"` (Blue) - Pumps, Valves, Motors.
- `"SENSOR"` (Purple) - Transmitters (Pressure, Flow, Temp).
- `"PID"` (Slate) - Control Loops.
- `"LOGIC"` (Amber) - PLCs, Interlocks, Decision Blocks.

- **`status`**: Visual health indicator.
- `"active"` (Green), `"inactive"` (Grey), `"fault"` (Red Pulse).

- **`meta`**: Contains static metadata and interactive widgets.

#### 2.2 The `meta` Object (Widgets)

Populate these fields to enable specific UI widgets inside the node.

| Widget         | Field         | JSON Structure                                |
| -------------- | ------------- | --------------------------------------------- |
| **Properties** | `properties`  | `[{"label": "Gain", "value": "0.5"}, ...]`    |
| **Info**       | `description` | `"Maintains system pressure at safe levels."` |

---

### 3. Edges Configuration

Edges define the signal flow between nodes.

| Field      | Type      | Description                                                      |
| ---------- | --------- | ---------------------------------------------------------------- |
| `id`       | `string`  | Unique ID (e.g., "e-1-2").                                       |
| `source`   | `string`  | ID of the starting node.                                         |
| `target`   | `string`  | ID of the receiving node.                                        |
| `type`     | `string`  | **Must be `"packetEdge"**` to enable the data packet animation.  |
| `label`    | `string`  | (Optional) Text badge shown on the line (e.g., "Set Point").     |
| `animated` | `boolean` | `true` for moving dashed lines (active flow), `false` for solid. |

---

### 4. Complete Example JSON

Save this as `system-config.json` to load the Dual Pump Logic system.

```json
{
	"nodes": [
		{
			"id": "PT-1000",
			"type": "custom",
			"position": { "x": 100, "y": 50 },
			"data": {
				"id": "PT-1000",
				"type": "SENSOR",
				"label": "Pipeline Pressure",
				"status": "active",
				"currentValue": 58.4,
				"meta": {
					"tagId": "PT-1000",
					"unit": "PSI",
					"description": "Monitors the main discharge pressure.",
					"minValue": 0,
					"maxValue": 150,
					"narrativeRef": "Primary feedback sensor.",
					"properties": [
						{ "label": "Hi-Hi Limit", "value": "85.0 PSI" },
						{ "label": "Lo-Lo Limit", "value": "20.0 PSI" }
					]
				}
			}
		},
		{
			"id": "PID-01",
			"type": "custom",
			"position": { "x": 400, "y": 50 },
			"data": {
				"id": "PID-01",
				"type": "PID",
				"label": "Pressure Loop",
				"status": "active",
				"currentValue": 58.4,
				"meta": {
					"tagId": "PID_01",
					"unit": "PSI",
					"narrativeRef": "Maintains 60 PSI Setpoint.",
					"properties": [
						{ "label": "Setpoint", "value": "60 PSI" },
						{ "label": "Output", "value": "85%" }
					]
				}
			}
		},
		{
			"id": "P-101",
			"type": "custom",
			"position": { "x": 100, "y": 450 },
			"data": {
				"id": "P-101",
				"type": "EQUIPMENT",
				"label": "Pump #1 (Lead)",
				"status": "active",
				"currentValue": 85,
				"meta": {
					"unit": "% Spd",
					"narrativeRef": "Running in Auto.",
					"properties": [{ "label": "Amps", "value": "12.4 A" }]
				}
			}
		}
	],
	"edges": [
		{
			"id": "link-1",
			"source": "PT-1000",
			"target": "PID-01",
			"type": "packetEdge",
			"label": "PV",
			"animated": true,
			"style": { "stroke": "#6366f1", "strokeWidth": 2 }
		},
		{
			"id": "link-2",
			"source": "PID-01",
			"target": "P-101",
			"type": "packetEdge",
			"label": "Speed Cmd",
			"animated": true,
			"style": { "stroke": "#10b981", "strokeWidth": 2 }
		}
	]
}
```

Note: While { x, y } coordinates are required for the schema validity, the Auto-Layout engine will calculate the final positions at runtime. You can set these to { "x": 0, "y": 0 } safely.
