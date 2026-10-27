import { useEffect, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import Overlay from "./Overlay";

export type DictationState = "idle" | "recording" | "processing";

/** `meterBars`: how many recent audio levels the overlay meter shows. */
export default function App({ meterBars = 18 }: { meterBars?: number }) {
  const [state, setState] = useState<DictationState>("idle");
  const [levels, setLevels] = useState<number[]>(() => Array<number>(meterBars).fill(0));

  useEffect(() => {
    const unlisten = listen<DictationState>("dictation-state", (event) => {
      setState(event.payload);
      if (event.payload === "recording") setLevels(Array<number>(meterBars).fill(0));
    });
    return () => {
      unlisten.then((fn) => fn());
    };
  }, [meterBars]);

  useEffect(() => {
    const unlisten = listen<number>("audio-level", (event) => {
      setLevels((prev) => [...prev.slice(1), event.payload]);
    });
    return () => {
      unlisten.then((fn) => fn());
    };
  }, []);

  return <Overlay state={state} levels={levels} />;
}
