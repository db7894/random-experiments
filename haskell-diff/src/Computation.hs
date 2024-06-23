-- inspired by https://blog.janestreet.com/computations-that-differentiate-debug-and-document-themselves/
-- https://www.danielbrice.net/blog/automatic-differentiation-is-trivial-in-haskell/

{- A computation involving some set of variables. It can be evaluated, and
    the partial derivative of each variable will be automatically computed. -}

module Computation
  ( Computation(..)
  , Variable(..)
  , VariableID(..)
  , Variableable(..)
  , constant
  , variable
  , sumComputation
  , square
  , evaluate
  ) where

import qualified Data.Map.Strict as Map
import Data.String (IsString(..))

newtype VariableID = VariableID String deriving (Eq, Ord, Show)

instance IsString VariableID where
  fromString = VariableID

data Variable = Variable
  { varId :: VariableID
  , varValue :: Double
  , varDerivative :: Double
  } deriving (Show)

data Computation
  = Constant Double
  | Var Variable
  | Sum [Computation]
  | Square Computation
  deriving (Show)

class Variableable v where
  createVariable :: VariableID -> Double -> v
  getValue :: v -> Double
  getDerivative :: v -> Double
  setValue :: v -> Double -> v

instance Variableable Variable where
  createVariable id val = Variable id val 0
  getValue = varValue
  getDerivative = varDerivative
  setValue var val = var { varValue = val }

-- Main functions
constant :: Double -> Computation
constant = Constant

variable :: Variable -> Computation
variable = Var

sumComputation :: [Computation] -> Computation
sumComputation = Sum

square :: Computation -> Computation
square = Square

evaluate :: Computation -> (Double, Map.Map VariableID Double)
evaluate comp = case comp of
  Constant c -> (c, Map.empty)
  Var v -> (getValue v, Map.singleton (varId v) 1)
  Sum cs -> 
    let (vals, derivs) = unzip (map evaluate cs)
    in (Prelude.sum vals, Map.unionsWith (+) derivs) -- Prelude is implicitly imported in every Haskell module
  Square c ->
    let (val, deriv) = evaluate c
    in (val * val, Map.map (* (2 * val)) deriv)
