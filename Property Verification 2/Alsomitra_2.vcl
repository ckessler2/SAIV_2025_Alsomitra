-- Adapted from: https://github.com/KatyaKom/DAIR-course/blob/main/Group-DAIR/Working_ADV_Training/alsomitra_5_mc.vcl
-- By Prab Singh, Ada Grecner, and Alexander Wickham

--------------------------------------------------------------------------------
-- Copied from: Full specification of the ACAS XU networks

--------------------------------------------------------------------------------
-- Inputs

type InputVector = Vector Rat 6

dv_x = 0
dv_y = 1
d_omega = 2
d_theta = 3
d_x = 4
d_y = 5

--------------------------------------------------------------------------------
-- Outputs

type OutputVector = Vector Rat 1

e_x = 0

--------------------------------------------------------------------------------
-- The network

-- Next we use the `network` annotation to declare the name and the type of the
-- neural network we are verifying. The implementation is passed to the compiler
-- via a reference to the ONNX file at compile time.

@network
alsomitra : InputVector -> OutputVector

--------------------------------------------------------------------------------
-- Normalisation

-- minimum and maximum are taken from training data. None of the inputs are scaled

type UnnormalisedInputVector = Vector Rat 6

-- Next we define the minimum and maximum values that each input can take.
-- These correspond to the range of the inputs that the network is designed
-- to work over. These are taken from the training dataset
minimumInputValues : UnnormalisedInputVector
minimumInputValues = [0.00,0.00,0.00,0.00,0.00,0.00]

maximumInputValues : UnnormalisedInputVector
maximumInputValues = [1.0,1.0,1.0,1.0,1.0,1.0]



validInput : UnnormalisedInputVector -> Bool
validInput x = forall i . minimumInputValues ! i <= x ! i <= maximumInputValues ! i

normalise: UnnormalisedInputVector -> InputVector
normalise x = foreach i .  x ! i


-- Inputs are not normalised, as required for CORA
norm_alsomitra : UnnormalisedInputVector -> OutputVector
norm_alsomitra x = alsomitra (normalise x)

@parameter
ystar: Rat

@parameter
epsilon: Rat

@parameter(infer=True)
n : Nat

@dataset
trainingInputs : Vector InputVector n

--------------------------------------------------------------------------------
--Epsilon balls around training point (input) definition
boundedByEpsilon : InputVector -> Bool
boundedByEpsilon perturbedInput = -epsilon <= perturbedInput ! dv_x <= epsilon and
	-epsilon <= perturbedInput ! dv_y <= epsilon and
	-epsilon <= perturbedInput ! d_omega <= epsilon and
	-epsilon <= perturbedInput ! d_theta <= epsilon and
	-epsilon <= perturbedInput ! d_x <= epsilon and
	-epsilon <= perturbedInput ! d_y <= epsilon



-- Property 1

-- If the drone is far above the line, the network will always make it pitch down (e_x > 0.5)
--Line equation: in nondimensional terms: y = -x
-- X normalisation: x_norm = (x - 0.482420778264618) / 41.232296222734604 
-- X denormalisation: x = (x_norm * 41.232296222734604 ) + 0.482420778264618
-- Y normalisation: y_norm = (y + 41.681405266088724) / 45.878638794593570
-- Y denormalisation: y = (y_norm * 45.878638794593570 ) - 41.681405266088724

--- Drone far above line; nondimensional y > 2 above line
droneFarAboveLine : UnnormalisedInputVector -> Bool
droneFarAboveLine x =
	(x ! d_y * 45.878638794593570 ) - 41.681405266088724 >= -((x ! d_x * 41.232296222734604 ) + 0.482420778264618) + ystar

droneFarBelowLine : UnnormalisedInputVector -> Bool
droneFarBelowLine x =
	(x ! d_y * 45.878638794593570 ) - 41.681405266088724 <= -((x ! d_x * 41.232296222734604 ) + 0.482420778264618) - ystar
  

--property1 : Bool
--property1 = forall x . validInput x and droneFarAboveLine x =>
--property1 = forall x . validInput x and droneFarAboveLine x=>
--  norm_alsomitra x ! e_x >= 0.5


--Local version
property1_ : InputVector -> Bool
property1_ input = forall pertubation .
  	let perturbedInput = input - pertubation in boundedByEpsilon pertubation and 
	validInput perturbedInput and droneFarAboveLine perturbedInput => 
	alsomitra perturbedInput ! e_x >= 0.5

@property
property1 : Vector Bool n
property1 = foreach i . property1_ (trainingInputs ! i)

--@property
--property2 : Bool

--property2 = forall x . validInput x and droneFarBelowLine x=>
--  norm_alsomitra x ! e_x <= 0.5

--Local version
property2_ : InputVector -> Bool
property2_ input = forall pertubation .
  	let perturbedInput = input - pertubation in boundedByEpsilon pertubation and 
	validInput perturbedInput and droneFarBelowLine perturbedInput => 
	alsomitra perturbedInput ! e_x <= 0.5

@property
property2 : Vector Bool n
property2 = foreach i . property2_ (trainingInputs ! i)