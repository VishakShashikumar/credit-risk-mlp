// CI check: predict.js must reproduce scikit-learn's probabilities for 200
// held-out applicants and give the same flag decision at the 0.29 cutoff.
const fs = require("fs");
const path = require("path");
const { predictProba } = require("../predict.js");

const root = path.join(__dirname, "..");
const model = JSON.parse(fs.readFileSync(path.join(root, "model.json"), "utf8"));
const { rows, sklearn_proba } = JSON.parse(
  fs.readFileSync(path.join(__dirname, "fixtures", "sample_rows.json"), "utf8"));

let worst = 0;
let flagMismatches = 0;
rows.forEach((row, i) => {
  const p = predictProba(model, row);
  worst = Math.max(worst, Math.abs(p - sklearn_proba[i]));
  if ((p >= model.threshold) !== (sklearn_proba[i] >= model.threshold)) flagMismatches++;
});

console.log(`Checked ${rows.length} applicants; largest gap vs scikit-learn = ${worst.toExponential(2)}`);
if (worst > 1e-9 || flagMismatches > 0) {
  console.error(`FAILED: gap ${worst}, flag mismatches ${flagMismatches}`);
  process.exit(1);
}
console.log("predict.js matches scikit-learn");
