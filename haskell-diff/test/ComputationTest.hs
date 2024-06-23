module Main where

import Test.HUnit
import qualified Computation as C
import qualified Data.Map.Strict as Map

testComputation :: Test
testComputation = TestCase $ do
  let x = C.createVariable (C.VariableID "x") 2 :: C.Variable
      y = C.createVariable (C.VariableID "y") 4 :: C.Variable
      comp = C.square (C.sumComputation [C.variable x, C.square (C.sumComputation [C.variable y, C.constant 1])])
      (result, derivatives) = C.evaluate comp
  assertEqual "Result" 729 result -- (2 + (4+1)^2)^2 = 27^2 = 729.
  assertEqual "Derivative of x" 54 (Map.findWithDefault 0 (C.VariableID "x") derivatives)
  assertEqual "Derivative of y" 540 (Map.findWithDefault 0 (C.VariableID "y") derivatives)

main :: IO Counts
main = runTestTT testComputation