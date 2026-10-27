import type { DictationState } from "./App";

export default function Overlay({
  state,
  levels,
  barWidth = 3,
  barGap = 3,
  height = 20,
  minBar = 3,
}: {
  state: DictationState;
  levels: number[];
  barWidth?: number;
  barGap?: number;
  height?: number;
  minBar?: number;
}) {
  if (state === "idle") return null;

  return (
    <div className="pill">
      {state === "recording" ? (
        <svg
          className="meter"
          role="meter"
          aria-label="Microphone level"
          aria-valuemin={0}
          aria-valuemax={1}
          aria-valuenow={levels[levels.length - 1]}
          width={levels.length * (barWidth + barGap) - barGap}
          height={height}
        >
          {[...levels.keys()].map((slot) => {
            const barHeight = minBar + (height - minBar) * levels[slot];
            return (
              <rect
                key={slot}
                x={slot * (barWidth + barGap)}
                y={(height - barHeight) / 2}
                width={barWidth}
                height={barHeight}
                rx={barWidth / 2}
              />
            );
          })}
        </svg>
      ) : (
        <>
          <span className="spinner" />
          <span className="label">Processing{"…"}</span>
        </>
      )}
    </div>
  );
}
