/-
  VeriMath Tactics
  ================
  Imports only the Mathlib modules needed for VeriMath's 7 template types.
  Keeping imports minimal reduces .olean compile time and RAM usage on Pi 4.

  Imported for:
  - norm_num    → arithmetic, definite integrals, geometry distance
  - ring        → algebraic simplification
  - omega       → linear integer arithmetic (equation solving)
  - linarith    → linear real arithmetic
  - simp        → general simplification (used in derivative template)
  - decide      → decidable propositions (primality for small n)
  - deriv_*     → derivative simp lemmas
  - IntervalIntegral → definite integrals via norm_num
-/

import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Linarith
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Deriv
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Analysis.SpecialFunctions.Log.Deriv
import Mathlib.MeasureTheory.Integral.IntervalIntegral.Basic
import Mathlib.MeasureTheory.Integral.IntervalIntegral.FundThmCalculus
import Mathlib.Data.Nat.Prime.Basic
import Mathlib.Geometry.Euclidean.Basic

-- Re-export everything for verification files
-- (verification temp files only need `import VeriMath.Tactics`)
