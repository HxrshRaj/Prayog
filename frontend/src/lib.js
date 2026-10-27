export async function api(path, options) {
  const r = await fetch(`/api${path}`, options);
  if (!r.ok) throw new Error(`${r.status} ${(await r.text()).slice(0, 200)}`);
  return r.json();
}

/** "0.8123 ± 0.0101" style formatting; `digits` decimals. */
export function fmt(stat, digits = 3) {
  if (!stat) return "–";
  return `${stat.mean.toFixed(digits)} ± ${stat.std.toFixed(digits)}`;
}

export const VERDICT_CLASS = {
  reproduced: "ok",
  "partly reproduced": "warn",
  "not reproduced": "bad",
  "not run": "muted",
};

/** summary {variant: {metric: {mean,std}}} -> rows for a bar chart of one metric. */
export function barRows(summary, metric, variants) {
  return (variants || Object.keys(summary || {}))
    .filter((v) => summary?.[v]?.[metric])
    .map((v) => ({ variant: v, mean: summary[v][metric].mean, std: summary[v][metric].std }));
}

/** curves {variant: {x, mean, std}} -> merged rows for recharts */
export function curveRows(curves) {
  const variants = Object.keys(curves || {});
  if (!variants.length) return [];
  return curves[variants[0]].x.map((x, i) => {
    const row = { x };
    for (const v of variants) {
      const m = curves[v].mean[i];
      const s = curves[v].std[i];
      row[v] = m;
      row[`${v}_band`] = [m - s, m + s];
    }
    return row;
  });
}
