import Lake
open Lake DSL

package VeriMath where
  version := v!"1.0.0"

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "stable"

@[default_target]
lean_lib VeriMath where
  roots := #[`VeriMath.Tactics]
