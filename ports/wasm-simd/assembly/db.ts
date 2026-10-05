// Transposed from appx/dsp at ec84edcd331af8810be9df42cf5d16f3a348d6d5 (2026-10-05).
/** Levels at or below this linear amplitude report as the floor. */
const FLOOR_AMPLITUDE: f64 = 1.0e-6;
/** 20·log10(FLOOR_AMPLITUDE). */
const FLOOR_DB: f64 = -120.0;

/** Converts a linear amplitude (1.0 = full scale) to dBFS, floored at -120 dB. */
export function linearToDb(amplitude: f64): f64 {
  return amplitude <= FLOOR_AMPLITUDE ? FLOOR_DB : 20.0 * Math.log10(amplitude);
}
