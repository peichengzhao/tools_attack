import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import { fileURLToPath } from "node:url";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));

function usage() {
  return [
    "Usage:",
    "  node generate_split_manifest.mjs --dataset-dir <data-dir> [--output-dir <output-dir>]",
    "  node generate_split_manifest.mjs <data-dir> [--output-dir <output-dir>]",
    "",
    "<data-dir> must contain test_cases_ds_base.json and test_cases_dh_base.json.",
    "The output directory defaults to the directory containing this script.",
  ].join("\n");
}

function parseArgs(args) {
  let datasetDir;
  let outputDir = scriptDir;

  for (let index = 0; index < args.length; index += 1) {
    const arg = args[index];
    if (arg === "--help" || arg === "-h") {
      console.log(usage());
      process.exit(0);
    }
    if (arg === "--dataset-dir" || arg === "-d") {
      datasetDir = args[index + 1];
      index += 1;
    } else if (arg === "--output-dir" || arg === "-o") {
      outputDir = args[index + 1];
      index += 1;
    } else if (arg.startsWith("-")) {
      throw new Error(`Unknown option: ${arg}\n\n${usage()}`);
    } else if (datasetDir === undefined) {
      datasetDir = arg;
    } else {
      throw new Error(`Unexpected argument: ${arg}\n\n${usage()}`);
    }
  }

  if (!datasetDir) {
    throw new Error(`Missing dataset directory.\n\n${usage()}`);
  }
  if (!outputDir) {
    throw new Error(`Missing output directory.\n\n${usage()}`);
  }

  return {
    datasetDir: path.resolve(datasetDir),
    outputDir: path.resolve(outputDir),
  };
}

const { datasetDir, outputDir } = parseArgs(process.argv.slice(2));
fs.mkdirSync(outputDir, { recursive: true });
const fieldOrder = [
  "Attack Type",
  "User Instruction",
  "User Tool",
  "Tool Parameters",
  "Attacker Instruction",
  "Attacker Tools",
  "Tool Response Template",
];

function sha256(value) {
  return crypto.createHash("sha256").update(value, "utf8").digest("hex");
}

function normalize(value) {
  if (typeof value === "string") return value.normalize("NFC");
  if (Array.isArray(value)) return value.map(normalize);
  if (value && typeof value === "object") {
    return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, normalize(item)]));
  }
  return value;
}

function canonicalCase(item) {
  return normalize(Object.fromEntries(fieldOrder.map((field) => [field, item[field]])));
}

function csvCell(value) {
  const text = String(value ?? "");
  return /[",\n\r]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function loadCases(fileName, prefix, attackType) {
  const filePath = path.join(datasetDir, fileName);
  const cases = JSON.parse(fs.readFileSync(filePath, "utf8"));
  return cases.map((item, index) => {
    const caseId = `${prefix}-${String(index + 1).padStart(4, "0")}`;
    const canonical = canonicalCase(item);
    const canonicalJson = JSON.stringify(canonical);
    const contentSha256 = sha256(canonicalJson);
    const splitKey = sha256(`AAAI27-S1-v1|${caseId}|${contentSha256}`);
    const attackerTools = item["Attacker Tools"];
    if (!Array.isArray(attackerTools) || attackerTools.length === 0) {
      throw new Error(`${caseId}: Attacker Tools is empty`);
    }
    return {
      case_id: caseId,
      source_file: `data/${fileName}`,
      source_index: index + 1,
      attack_type: attackType,
      user_tool: item["User Tool"],
      target_tool: attackerTools[0],
      template_sha256: sha256(item["Tool Response Template"]),
      content_sha256: contentSha256,
      split_key: splitKey,
      canonical_json_sha256: sha256(canonicalJson),
    };
  });
}

const cases = [
  ...loadCases("test_cases_ds_base.json", "DS", "data-stealing"),
  ...loadCases("test_cases_dh_base.json", "DH", "direct-harm"),
];

const splitCounts = {
  "data-stealing": [326, 109, 109],
  "direct-harm": [306, 102, 102],
};

for (const attackType of Object.keys(splitCounts)) {
  const stratum = cases.filter((item) => item.attack_type === attackType).sort((a, b) => a.split_key.localeCompare(b.split_key));
  const counts = splitCounts[attackType];
  if (stratum.length !== counts.reduce((sum, count) => sum + count, 0)) {
    throw new Error(`${attackType}: unexpected case count ${stratum.length}`);
  }
  let offset = 0;
  for (const [split, count] of [["train", counts[0]], ["validation", counts[1]], ["test", counts[2]]]) {
    for (const item of stratum.slice(offset, offset + count)) item.split = split;
    offset += count;
  }
}

const ordered = cases.slice().sort((a, b) => a.case_id.localeCompare(b.case_id));
const headers = [
  "case_id",
  "source_file",
  "source_index",
  "attack_type",
  "split",
  "user_tool",
  "target_tool",
  "template_sha256",
  "content_sha256",
  "canonical_json_sha256",
  "split_key",
];
const csv = [headers.join(","), ...ordered.map((item) => headers.map((header) => csvCell(item[header])).join(","))].join("\n") + "\n";
fs.writeFileSync(path.join(outputDir, "split_manifest.csv"), csv, "utf8");

for (const split of ["train", "validation", "test"]) {
  const ids = ordered.filter((item) => item.split === split).map((item) => item.case_id).join("\n") + "\n";
  fs.writeFileSync(path.join(outputDir, `${split}_case_ids.txt`), ids, "utf8");
}

const sourceHashes = [
  ["data/test_cases_ds_base.json", sha256(fs.readFileSync(path.join(datasetDir, "test_cases_ds_base.json")))],
  ["data/test_cases_dh_base.json", sha256(fs.readFileSync(path.join(datasetDir, "test_cases_dh_base.json")))],
];
const manifestSha = sha256(csv);
const provenance = [
  "Manifest version: AAAI27-S1-v1",
  "Canonical JSON: UTF-8 JSON.stringify over the seven fields in fieldOrder, with recursive NFC normalization; no whitespace.",
  "Stable IDs: official base-file order, 1-based, DS-0001..DS-0544 and DH-0001..DH-0510.",
  "Target tool: Attacker Tools[0] (the S1 target in the evaluation code).",
  "Split: sort SHA256(AAAI27-S1-v1|case_id|content_sha256) independently within each attack stratum; 60/20/20 slices.",
  `Manifest SHA-256: ${manifestSha}`,
  ...sourceHashes.map(([file, hash]) => `${file} SHA-256: ${hash}`),
  "",
].join("\n");
fs.writeFileSync(path.join(outputDir, "split_manifest_provenance.txt"), provenance, "utf8");

console.log(JSON.stringify({
  total: ordered.length,
  byAttackType: Object.fromEntries(Object.keys(splitCounts).map((attackType) => [attackType, ordered.filter((item) => item.attack_type === attackType).length])),
  bySplit: Object.fromEntries(["train", "validation", "test"].map((split) => [split, ordered.filter((item) => item.split === split).length])),
  manifestSha256: manifestSha,
}, null, 2));
