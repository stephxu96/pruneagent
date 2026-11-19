import { describe, expect, it } from "vitest";
import { changedLayers, changesText, duration, durationRange, lossText, qualityLoss, timeLeftText } from "./lib";

describe("plain-language wording", () => {
  it("groups changes by kind and counts layers from 1", () => {
    expect(changesText([{ layer: 2, strategy: "narrow_mlp" }, { layer: 3, strategy: "narrow_mlp" }])).toBe("slim layers 2, 3");
    expect(changesText([{ layer: 13, strategy: "drop_layer" }])).toBe("remove layer 13 entirely");
    expect(changesText([{ layer: 4, strategy: "remove_attention_group" }, { layer: 22, strategy: "narrow_mlp" }])).toBe(
      "remove an attention unit from layer 4; slim layer 22",
    );
  });

  it("says quality loss as a percentage, or 'times worse' when large", () => {
    expect(lossText(qualityLoss(16.1, 13.15))).toBe("22% worse");
    expect(lossText(qualityLoss(427.3, 12.66))).toBe("34× worse");
    expect(lossText(0)).toBe("no loss");
  });
});

describe("changedLayers", () => {
  it("finds the layers that differ between two steps", () => {
    const a = [{ mlp: 3072, groups: 8, dropped: false }, { mlp: 3072, groups: 8, dropped: false }];
    const b = [{ mlp: 3072, groups: 8, dropped: false }, { mlp: 3072, groups: 8, dropped: true }];
    expect([...changedLayers(a, b)]).toEqual([1]);
    expect(changedLayers(undefined, b).size).toBe(0);
  });
});

describe("time wording", () => {
  it("gives a range, and warns when the round limit comes first", () => {
    expect(durationRange(57, 76)).toBe("55 minutes–1 h 15 min");
    expect(duration(119)).toBe("2 h");
    expect(duration(57)).toBe("55 minutes");
    expect(timeLeftText({ minutes: [14, 14], worst_minutes: 14, rounds_left: 5, mb_left: 62 })).toBe(
      "Likely to stop at the round limit before the goal (5 rounds, about 15 minutes left)",
    );
    expect(timeLeftText({ minutes: [20, 30], worst_minutes: 60, rounds_left: 20, mb_left: 40 })).toBe(
      "About 20–30 minutes left, at most 1 h (20 rounds left)",
    );
  });
});
