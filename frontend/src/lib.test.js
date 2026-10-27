import { describe, expect, it } from "vitest";
import { barRows, curveRows, fmt } from "./lib.js";

describe("lib", () => {
  it("formats mean ± std", () => {
    expect(fmt({ mean: 0.81234, std: 0.0101 })).toBe("0.812 ± 0.010");
    expect(fmt(undefined)).toBe("–");
  });

  it("builds bar rows only for variants having the metric", () => {
    const s = { a: { acc: { mean: 1, std: 0.1 } }, b: {} };
    expect(barRows(s, "acc")).toEqual([{ variant: "a", mean: 1, std: 0.1 }]);
  });

  it("merges curves into recharts rows with a spread band", () => {
    const rows = curveRows({ a: { x: [0, 10], mean: [1, 3], std: [0.5, 1] } });
    expect(rows[1]).toEqual({ x: 10, a: 3, a_band: [2, 4] });
  });
});
