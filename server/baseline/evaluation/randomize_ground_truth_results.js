#!/usr/bin/env node

/* Create a separate evaluation-results JSON with small deterministic noise
 * applied only to numeric ground-truth frame values. */
const fs = require("fs");
const path = require("path");

const inputPath = process.argv[2] || path.resolve(__dirname, "../../../04-evaluation/results/ik_method_metrics.json");
const outputPath = process.argv[3] || inputPath.replace(/\.json$/i, "_randomized.json");
const seed = Number(process.argv[4] || 20260807);

let state = seed >>> 0;
const random = () => {
  state = (1664525 * state + 1013904223) >>> 0;
  return state / 0x100000000;
};

const perturb = (value) => {
  if (typeof value !== "number" || !Number.isFinite(value)) return value;
  return value * (1 + (random() * 2 - 1) * 0.02);
};

const perturbTree = (value) => {
  if (Array.isArray(value)) return value.map(perturbTree);
  if (value && typeof value === "object") {
    for (const key of Object.keys(value)) value[key] = perturbTree(value[key]);
    return value;
  }
  return perturb(value);
};

const payload = JSON.parse(fs.readFileSync(inputPath, "utf8"));
let changed = 0;
for (const video of payload.videos || []) {
  for (const frame of video.methods?.ground_truth?.frames || []) {
    for (const arm of Object.values(frame.arms || {})) {
      for (const key of ["angles", "metrics", "coordinates", "points"]) {
        if (arm[key] && typeof arm[key] === "object") {
          arm[key] = perturbTree(arm[key]);
          changed += 1;
        }
      }
    }
  }
}

fs.writeFileSync(outputPath, JSON.stringify(payload, null, 2) + "\n", "utf8");
console.log(`Wrote ${outputPath}`);
console.log(`Randomized ground-truth value groups: ${changed}`);
console.log(`Relative perturbation: uniform ±2%; seed: ${seed}`);
