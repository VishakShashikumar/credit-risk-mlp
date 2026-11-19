/*
 * Runs the Module 8 network in the browser from the numbers in model.json.
 * Mirrors the scikit-learn pipeline: merge undocumented codes, one-hot encode
 * sex/education/marital status, standardize the other 20 columns, then two
 * ReLU layers and a logistic output. Also loadable from Node for CI tests.
 */
(function (root) {
  "use strict";

  function clean(applicant) {
    const a = Object.assign({}, applicant);
    if ([0, 5, 6].includes(a.EDUCATION)) a.EDUCATION = 4;
    if (a.MARRIAGE === 0) a.MARRIAGE = 3;
    return a;
  }

  function encode(model, applicant) {
    const a = clean(applicant);
    const x = [];
    for (const [col, cats] of Object.entries(model.nominal)) {
      for (const c of cats) x.push(a[col] === c ? 1 : 0);
    }
    model.scaled_columns.forEach((col, i) => {
      x.push((a[col] - model.mean[i]) / model.scale[i]);
    });
    return x;
  }

  function predictProba(model, applicant) {
    let act = encode(model, applicant);
    const last = model.layers.length - 1;
    model.layers.forEach((layer, li) => {
      const out = layer.bias.slice();
      for (let i = 0; i < act.length; i++) {
        const xi = act[i];
        if (xi === 0) continue;
        const row = layer.weights[i];
        for (let j = 0; j < out.length; j++) out[j] += xi * row[j];
      }
      act = li < last ? out.map((v) => Math.max(v, 0)) : out.map((v) => 1 / (1 + Math.exp(-v)));
    });
    return act[0];
  }

  const api = { predictProba, encode };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.CreditModel = api;
})(typeof window !== "undefined" ? window : globalThis);
