# **Neural Network Verification for Gliding Drone Control: A Case Study** 

Colin Kessler<sup>1</sup><sup>_,_2</sup> �, Ekaterina Komendantskaya<sup>1</sup> , Marco Casadio<sup>1</sup> , Ignazio Maria Viola<sup>2</sup> , Thomas Flinkow<sup>3</sup> , Albaraa Ammar Othman<sup>1</sup> , Alistair Malhotra<sup>1</sup> , and Robbie McPherson<sup>1</sup> 

- 1 Heriot-Watt University and Edinburgh Centre for Robotics, UK `ck2049@hw.ac.uk` 

> 2 School of Engineering, University of Edinburgh, UK 

- 3 Maynooth University, Maynooth, Ireland 

**Abstract.** As machine learning is increasingly deployed in autonomous systems, verification of neural network controllers is becoming an active research domain. Existing tools and annual verification competitions suggest that soon this technology will become effective for realworld applications. Our application comes from the emerging field of microflyers that are passively transported by the wind, which may have various uses in weather or pollution monitoring. Specifically, we investigate centimetre-scale bio-inspired gliding drones that resemble _Alsomitra macrocarpa_ diaspores. In this paper, we propose a new case study on verifying _Alsomitra_ -inspired drones with neural network controllers, with the aim of adhering closely to a target trajectory. We show that our system differs substantially from existing VNN and ARCH competition benchmarks, and show that a combination of tools holds promise for verifying such systems in the future, if certain shortcomings can be overcome. We propose a novel method for robust training of regression networks, and investigate formalisations of this case study in Vehicle and CORA. Our verification results suggest that the investigated training methods do improve performance and robustness of neural network controllers in this application, but are limited in scope and usefulness. This is due to systematic limitations of both Vehicle and CORA, and the complexity of our system reducing the scale of reachability, which we investigate in detail. If these limitations can be overcome, it will enable engineers to develop safe and robust technologies that improve people’s lives and reduce our impact on the environment. 

**Keywords:** Neural Network Control · Bioinspired Robots · Verification of Cyber-Physical Systems · Machine Learning. 

## **1 Introduction** 

A recent research trend in drone design concerns the development of gliding microdrones, which could serve a function as airborne sensors and remain aloft for extended periods of time [16, 18, 21, 35]. Current research focuses on the 

2 C. Kessler et al. 

aerodynamics of seeds that have exceptional wind dispersal mechanisms such as, for example, the _Taraxacum_ (dandelion) [9] and _Alsomitra_ (Javan cucumber). This case study focuses specifically on _Alsomitra_ -inspired drones (Figs. 1,4) as the aerodynamics underlying the flight of this diaspore is unique in the plant kingdom, enhancing the dispersal mechanism provided by the wind by an efficient gliding flight. This allows one of the heaviest seeds (314 mg) [4] to reach a similar descent velocity than some of the lightest seeds such as the dandelion (0.6 mg) [9]. Because of this unique feature, several authors have considered this diaspore as a bioinspiration for microdrones [26,35]. Such drones could function as distributed sensors in the atmosphere, for weather monitoring or detecting pollutants [16, 18,21,26,35]. This could be particularly useful for environmental monitoring and meteorology, with research and regulations moving towards incorporating drone observations to improve weather predictions [11, 36]. It has been demonstrated that such systems are capable of sustained flight with active control and internal electronics [18], although more work on effective actuation and control methods is needed in the future. 

Neural networks (NNs) have been widely investigated for drone control, for both quadcopters [3, 25] and fixed-wing designs [31, 33]. The control of small passive gliders is a relatively unexplored field, with the most relevant works involving larger aircraft [1, 33] or without continuous control [18]. For our application, we consider NN control since it has been shown to achieve accurate and robust control for systems with uncertain dynamics [17], it is particularly applicable to controlling swarms [30], and improvements to low-order aerodynamic modelling [24] facilitates easier simulations of such drones. This approach could facilitate particularly lightweight and low-cost drones - such as with analogue network circuits printed on flexible substrates acting as the body of the gliding drone [28, 32]. One could alternatively consider uncontrolled flying sensors [16,21,35] or traditional approaches such as state-space or model predictive control. However, one should consider that such systems will need to be verifiably safe with regard to people, other air users, and the environment [36]. These drones could collide with each other, veer into unsafe airspace, fall into an endangered ecosystem, or otherwise cause harm. The utility of uncontrolled flyers would be hampered by such issues, unless they can be made biodegradable. Traditional control methods may be applied, but NN methods have advantages in that they can be made data-driven and adaptive, and printed NN circuits could lead to lighter designs than digital microcontrollers. 

### **1.1 Contributions** 

Our first aim is the introduction of a novel case study (outlined in Sect. 4) in the verification of _Alsomitra_ -inspired drone controllers (our modelling methods are explained in Sect. 3), that differs significantly from existing benchmarks. Unlike VNN-COMP benchmarks such as ACAS Xu, our study involves regression control and continuous dynamic equations. Compared to ARCH-COMP benchmarks such as QUAD, our system involves differential equations that are far more complex in terms of the number of non-linear terms. Moreover, unlike the 

Neural Network Verification for Gliding Drone Control: A Case Study 

3 



**Fig. 1.** An artist’s impression of a swarm of gliding drones inspired by _Alsomitra_ seeds [7]. 

majority of ARCH-COMP cases, this problem does not have as natural a notion of the start, goal, safe, and unsafe states; and thus requires an out-of-the-box approach to property specification. 

We propose our ideal formalisation of the problem in Sect. 4.1, and distil the formalisation down to properties that can be handled with available tools (Marabou [20] implemented with Vehicle [10], and CORA [2]) in Sects. 7 and 8. The choice is motivated by the fact that each can be seen as a representative of a set of tools that come from the research communities of VNN-COMP [5] and ARCH-COMP [15], respectively. We present a new implementation of adversarial training for Lipschitz robustness applied to regression training for our controllers in Sect. 5, and present the results of verifying those properties with our robust networks in Sects. 7.3 and 8.2. 

Our second aim is to present the lessons learnt from investigating this case study, to help inform the development of relevant tools for similar real-life cyberphysical projects in the future (Sect. 9). The main lesson learnt is that no single existing tool ticks all the desirable boxes. Moreover, each individual tool we chose would benefit from further development in several aspects that are crucial for real-life models. Concretely: 

- On the Vehicle side, the verification properties that arise in the presented study are more complex than the usual VNN-COMP benchmarks in at least three ways. 

   - Firstly, the constraints on the input vector are more complex: instead of constraining individual vector elements by constants (as e.g. in _a ≤ xi ≤ b_ ), as is the case in the majority of benchmarks including ACAS XU [19], the constraints establish relation between different vector elements, as e.g. in _xi ≤ cxj_ . This changes mathematical interpretation of the verification problem: it no longer boils down to defining a hyper-rectangle (or other constant shape) on the input space and propagating it through the network layers, but gives a more general case of linear programming that works on arbitrary input space constraints. Not every VNN-COMP [5] verifier will be able to deal with such verification properties: Marabou is one of the most general tools in this family of tools and this case study suggests this generality may play a bigger role in the future. 

- 4 C. Kessler et al. 

   - Secondly, for verification of Lipschitz robustness, we implemented _relational properties_ , i.e. properties that compare different outputs of a neural network. These properties are not natively supported by Marabou or Vehicle yet, and required some additional plumbing. On-going implementation of support for relational verification in Marabou will be useful for cases such as this. 

   - Finally, some novelty of our verification approach is derived from the fact that, unlike most benchmarks in VNN-COMP, our models are regression models, rather than verification models. Some of the methods for training and verification are specialised to classification tasks only, and we predict that this has to change with occurrence of new engineering-inspired benchmarks. 

- On the CORA side, the system outlined in this study required several workarounds in order to compute reachability: 

   - The complexity of our system of equations [24] far exceeds that of all ARCH-COMP [15] benchmarks, in the number of non-linear terms. This would cause the Jacobian and Hessian matrices to far exceed the maximum number of terms supported by MATLAB , and fail to run. The equations were simplified (Sect. 3.2) by constraining the pitch angle and using an angle-of-attack definition, solving the complexity issue, but (for any reasonably large initial set) the reachable set still tended to expand exponentially after relatively few timesteps. This was solved by dividing the initial set into smaller subsets, computing reachable sets for each, and combining the results. 

   - CORA expects a NN controller that takes the system variables as inputs, with relatively few layer types supported [2]. Certain parameters occupy wider ranges than others (for example, _θ ∈_ [ _−_ 0 _._ 93 _, −_ 0 _._ 07], _x ∈_ [0 _._ 48 _,_ 41 _._ 7]) but unlike Vehicle, input normalisation (keeping all inputs between 0 and 1) is not supported. This is problematic since we intend to observe the effect of adversarial training, for which the input ranges should to be normalised, such that PGD attacks occur in _ϵ_ ranges that are not imbalanced between input dimensions. A workaround was found by training an adversarial network on normalised data, then implementing normalisation layers to the start and end of the network. 

   - Unlike similar ARCH-COMP benchmarks such as QUAD, our notion of a goal region is less obvious. We want the drone to adhere to the target trajectory in _x_ and _y_ , so define a goal state as a region around that trajectory. 

Although this study considers only one modification of gliding drones, most of the paper’s conclusions will be common between their different modifications, such as e.g. dandelion-inspired drones, and the lessons learnt can be broadly applied to other continuous control tasks. All relevant files are publicly available here. 

Neural Network Verification for Gliding Drone Control: A Case Study 

5 

## **2 Background** 

### **2.1 Neural Network Control** 

For our control method, we will use the common closed-loop negative feedback method, an overview of which can be seen in Fig. 2. In simple terms, the controller in a drone is given information about its current state (such as position, relative to some desired state) as input, and outputs a command to an actuator which affects how the drone flies. The controller can be considered as an equation linking the system states to an actuation force that changes the states over time according to the system dynamics, where the controller design affects how the drone behaves. If a traditional control theory approach is difficult (such as if the dynamics are highly complex) or a data-driven approach is desireable (if collecting data is easier than modelling the system, or if adaption based on new data is required), an engineer might consider implementing a NN controller. 



**Fig. 2.** Overview of a negative feedback control system. For each control iteration, an error signal is calculated by subtracting the current system state (feedback) from the desired system state (input). A controller computes an actuation based on this error, which is applied to a simulated or real system (plant), resulting in some new output state. 

### **2.2 Verification Tools** 

The case study will rely on the following three groups of neural network verification (NNV) tools. The first group concerns verification of infinite time-horizon properties of controllers in isolation from verification of the overall system dynamics. The most famous benchmark in the domain is ACASXu, and the representative verifier is Marabou [20]; other tools, such as ERAN [27], Pyrat [23] or _αβ_ -CROWN [37] could be interchangeably used for the verification tasks in which Marabou is deployed in this paper; we refer the reader to VNN-Comp [5] for an in depth discussion of existing tools in this category. In addition, we use Vehicle [10], a higher-level interface on top of Marabou, and take advantage of its facility in bridging the _embedding gap_ [8] between the physical domains and vector representation of data. 

The second group of methods considers the neural controller together with the overall system dynamics to ensure that the entire system avoids unsafe states, see Fig. 3. This class of problems is also known under the umbrella term reachability verification and representative examples include e.g. POLAR-Express [34], 

6 C. Kessler et al. 



**Fig. 3.** General form of reachability specifications - dots represent the system at successive control time steps, and arrows represent the continuous trajectory of the system. Any trajectory starting in the initial set should never intersect an unsafe set, and always finish in the goal set. 

and CORA [2], see [25] for an exhaustive overview of the mainstream tools in this category. Representative benchmarks include simple dynamic problems such as the inverted pendulum, and more complex problems such as the quadcopter, space docking, and 2-wheeled obstacle avoidance. Each benchmark has a predetermined set of dynamic equations and a NN controller, with a mix of supervised learning (through behaviour-cloning) and reinforcement learning. The limitations of these benchmarks are in the complexity of the networks (large networks require reduction methods), complexity of the equations (systems are either linear, or relatively simple non-linear differential equations), and verification of complex properties (no tools can successfully verify the Spacecraft Docking benchmark as of the most recent results [15]). 

Finally, an important group of methods for practical NNV cases comes from machine learning domain, under the umbrella term of _property-driven training (PDT)_ . These methods allow to optimise a given neural network for satisfying a desired verification property, with a view of improving the verification success [6,12,14]. Although methods in this group vary, they usually deploy a form of training with projected gradient descent (PGD) [22]. PGD methods involve finding the worst-case perturbed example in a region around a data point, which can then be implemented as a loss function during training: 



where _θ_ represents the parameters of the NN; ( _x, y_ ) _∼D_ are input-label pairs sampled from the data distribution _D_ ; _E_ is the expected value, averaging the loss over all samples in the data distribution _D_ ; _δ ∈ ∆_ is the adversarial perturbation constrained within a feasible set _∆_ (e.g., _∥δ∥p ≤ ϵ_ ) and _L_ is the loss function (e.g., RMSE, MAE) measuring the discrepancy between the predicted output _fθ_ ( _x_ + _δ_ ) and the true label _y_ . 

The inner maximisation, which identifies the _worst-case_ adversarial perturbation _δ ∈ ∆_ , is performed using PGD that iteratively adjusts _δ_ by ascending the gradient of the loss function with respect to the input, followed by projection back onto the feasible set _∆_ (e.g., ensuring _∥δ∥p ≤ ϵ_ ). 

The outer minimisation, aimed at optimising the neural network parameters _θ_ to minimise the adversarial loss, is achieved using gradient descent. 

Neural Network Verification for Gliding Drone Control: A Case Study 

7 

## **3 Modelling Methodology** 

### **3.1** **_Alsomitra macrocarpa_** 

A dynamics model (Fig. 4) was derived from [24] resulting in a system of equations for falling plates with displaced centre of mass (CoM), as defined in Sect. 3.2. Based on experimental measurements, our model accurately describes the falling trajectories of _Alsomitra_ seeds by inferring aerodynamic forces from the angle of attack [24]. The flight characteristics are highly dependant on the CoM displacement ( _ex_ , Fig. 4), providing us with a convenient actuation method for an _Alsomitra_ -inspired drone. 



<!-- Start of picture text -->
44 4 4<br><!-- End of picture text -->

**Fig. 4.** (a) An _Alsomitra_ seed [7]. (b) A two-dimensional approximation of an _Alsomitra_ seed, with centre of mass (CoM) displaced by _ℓ_ CM (nondimensional form _ex_ = _ℓ_ CM _/ℓ_ ). (c) Effect of various _ex_ on gliding trajectories; according to a quasi-steady aerodynamic model ( [24], Sect. 3.2). As the CoM is displaced the trajectory behaviour is affected significantly. 

### **3.2 Equations** 

The following equations describe falling plates with a displaced centre of mass [24], with six system variables ( _x_ 1 _..._ 6, Equations 11 ... 16), involving mechanical ( _ℓ_ , _m_ , _g_ , _ρf_ , _I_ ) and aerodynamic ( _CCP_<sup>0,</sup><sup>_C_</sup> _CP_<sup>1,</sup><sup>_C_</sup> _CP_<sup>2,</sup><sup>_C_</sup> _L_<sup>1,</sup><sup>_C_</sup> _L_<sup>2,</sup><sup>_C_</sup> _D_<sup>0,</sup><sup>_C_</sup> _D_<sup>1,</sup><sup>_C_</sup> _D_<sup>_π/_2,</sup><sup>_CR_,</sup> _α_ 0, _δ_ ) constants chosen to match that of _Alsomitra_ seeds [7]. Several intermediate terms are included for simplicity (Equations 1 ... 10)., and a more detailed overview can be seen in Appendix 10.1 







8 C. Kessler et al. 















_mx_ ˙1 = � _m_ + _πρf ℓ_<sup>2</sup> _/_ 4� _x_ 3 _x_ 2 _−_ ( _πρf ℓ_<sup>2</sup> _/_ 4) _x_ 3<sup>2</sup><sup>_ℓCM_+</sup><sup>_L_</sup> _T_<sup>_x′_+</sup><sup>_Lx_</sup> _R_<sup>_′_+</sup><sup>_Dx′−m′g_sin</sup><sup>_x_4(11)</sup> 











Neural Network Verification for Gliding Drone Control: A Case Study 

9 

## **4 Verification Task** 

Our _Alsomitra_ model from Sect. 3.2 is used as the basis of a feedback control system with a NN controller, as described in Sect. 2.1. In our case, the plant is the aerodynamic model, and the desired input is a linear reference trajectory in _x_ 5 and _x_ 6 (translational _x_ and _y_ ): 



The feedback signal consists of the six system states, and the CoM displacement is actuated by a controller aiming to follow the target trajectory (Fig. 5). 



**Fig. 5.** As a control problem, we consider an _Alsomitra_ -inspired microdrone and attempt to follow a linear trajectory in two dimensions. 

As per the ARCH-COMP airplane and pendulum benchmarks [15], the neural network controller is trained using behaviour cloning. All simulations ran for a total of 20 s, with a model timestep of 0.01 s and a control timestep of 0.5 s. A PID controller actuates _y_ 1 ( _ex_ ) based on an error in _x_ 6, and the gains are tuned manually until the control system performs well for a range of starting _x_ 6 positions. For each controller actuation (24 per simulation, for nine simulations), the system states, _x_ 6 error, and PID actuation are recorded for use as training data. This data is imported to Python for standard regression learning, and networks are exported in .onnx format for evaluation (Fig. 6) and verification. All networks have 6 inputs, 3 hidden layers with 6, 4, and 1 nodes respectively with ReLU activation functions, and 1 output. 

### **4.1 Formalisation** 

The core of this case study lies in examining the challenges in adopting the existing NNV methods in this new domain. Our ideal formalisation of the problem would be as follows. We consider a hybrid program where the six system states _x_ 1 _, ..., x_ 6 change over continuous time _t_ according the dynamics model shown in Sect. 3.2, and a NN controller acts to change the system state discretely every 0.5 s. For any starting state _x_ 1 _, ..., x_ 6(0), after some time _t_<sup>_∗_</sup> the trajectory of the drone will always be within some small distance _y_<sup>_∗_</sup> of the target trajectory (ideally, _x_ 6 = _−x_ 5). This boils down to the following ideal verification property: 

10 C. Kessler et al. 



**Fig. 6.** PID and basic NN controller performance on an _Alsomitra_ -inspired drone. The naive network is trained on regression data obtained from simulations with the PID controller, and the resulting performance is similar but not perfect. 



There are several features that distinguishes this system from standard NNV benchmarks, and we aim to explain the technical implications of these challenges for existing verification technologies, and propose ways in which these challenges can be overcome: 

1. The system dynamics are continuous, therefore unlike standard control verification benchmarks (such as ACAS Xu [5]), control is modelled as a regression task as oppose to classification. 

2. Unlike the ARCH-COMP benchmarks [25] that have a pre-defined notion of safe and unsafe state, these gliding drones do not have a notion of safety in the sense of a pre-defined coordinate region. A safe state is instead defined in a relational way, as adhesion to certain safe trajectory. 

3. Unlike many ARCH-COMP benchmarks, our verification task requires modelling with an infinite time horizon. Each drone could stay airborne for an arbitrary duration of time, depending on the surrounding airflow. 

4. As defined by our model, the dynamics of gliding drones are more complex than what is currently handled by the ARCH-COMP benchmarks and tools, and in particular it is more complex than the dynamics handled by tools that can verify infinite-time horizon systems, such as KeyMaeraX [29]. 

The available verification tools (Sect. 2.2) do not allow us to formalise this idealised goal directly, since CORA does not support infinite time, and Marabou does not support differential equations. As a result, we simplified this general task as two simpler tasks (in the first case sacrificing the analysis of the overall system dynamics, and in the second case the infinite-time horizon and relational notion of the target state): 

Neural Network Verification for Gliding Drone Control: A Case Study 11 

1. The NN will never command the drone to deviate significantly from the target trajectory. This task was implemented in Marabou, using the Vehicle specification language since it facilitates complex property definition. 

2. Given an interval of initial positions and a finite time horizon, the NNcontrolled drone will always reach a goal region, defined as a region around the target trajectory within this finite time frame. This task was verified in CORA. 

We note that task 1 resembles in some way a robustness property [6], except for now we deal with a regression NN and robustness relative to a line rather than a given data point. 

## **5 Robustness Training for Regression** 

Since robustness is critically important for drone safety, it seems reasonable to attempt a form of adversarial training based on PGD methods for our controller. The guiding hypothesis was that a general improvement in NN robustness should lead to improved verification performance. Since our neural network is a regression model, the classification-based training methods surveyed in Sect. 2.2 could not easily be used without modification. We therefore had to modify the PGD algorithm to use an RMSE loss function instead of cross-entropy, and modify other aspects of the adversarial training algorithm that relied on the presence of discrete classes. 

We focused on two notion of robustness, _standard_ and _Lipschitz_ robustness [6]. Given _x_<sup>_∗_</sup> _∈D_ and constants _ϵ, δ, L ∈ R_ , 





Since the latter has been proven to be strictly stronger than the former in [6], we implemented a form of PGD training with a Lipschitz loss function. During each training epoch, the algorithm finds the worst-case adversarial example ( _x_<sup>_∗_</sup> , _f_ ( _x_<sup>_∗_</sup> )) in an _ϵ_ -ball around each training point ( _x_ , _f_ ( _x_ )).To optimise the regression model for Lipschitz robustness, we dynamically compute the highest value of _L_ from the training and adversarial points (according to Eq. 20), which is summed to the training RMSE loss, penalising the network for large gradients about each data point. This is expected to result in a network with a smoother and therefore more robust output, at the expense of some accuracy. 

## **6 Property-Driven Training (PDT)** 

A different PDT method using differentiable logics was independently developed and applied to this case study in a separate submission [13]. As that submission does not include verification experiments, we will evaluate several models optimised for Properties 1, 2, 4, and 5 in [13] for the sake of comparison. The 

12 C. Kessler et al. 

models are listed in Table 1. In the table, _adversarial_ results represent just by one model trained according to Sect. 5; but _DL2_ and _Gödel Logic_ models are optimised specifically for Properties 1, 2, 4, and 5 respectively. A value of _y_<sup>_∗_</sup> = 2 was chosen for properties 1, 2, and 4 in training, in order to keep the properties relatively strict whilst avoiding counterexamples in the dataset. 

## **7 Vehicle Implementation** 

Task 1 was broken down into five simpler specifications to be implemented in Vehicle (a detailed introduction to which can be found in [10]), to ensure the NNs control the drone as desired in various ways. Global properties relating to the controller’s output relative to the target trajectory are introduced in Sect. 7.1, and a local robustness property is introduced in Sect. 7.2. Global properties are verified for all inputs bounded by the training data (representing the entire parameter space over which our controller is trained), and the local property is evaluated about _ϵ_ -balls from the traiing data. 

### **7.1 Global Property Specifications** 

Our first goal is to ensure the controller never causes the drone to deviate from the target trajectory. To establish a performance criteria, properties 1-4 include _y_<sup>_∗_</sup> (a threshold distance from the target trajectory, Eqs. 17, 18), such that a critical _y_<sup>_∗_</sup> can be found per network per property where verification succeeds. For example, for properties 1 and 2, a lower critical _y_<sup>_∗_</sup> would indicate a controller that better adheres to the target trajectory: 

1. _If the drone is above the line by some threshold y_<sup>_∗_</sup> _, the NN output will always make the drone pitch down_ (Listing 1) 

   - _x_ 6 _≥−x_ 5 + _y_<sup>_∗_</sup> _⇒ f_ ( _x_ ) _≥_ 0 _._ 187 (21) 

2. _If the drone is below the line by some threshold y_<sup>_∗_</sup> _, the NN output will always make the drone pitch up_ 

   - _x_ 6 _≤−x_ 5 _− y_<sup>_∗_</sup> _⇒ f_ ( _x_ ) _≤_ 0 _._ 187 (22) 

Our third property is reversed, where a larger _y_<sup>_∗_</sup> would indicate better adherence to a larger region around the target trajectory: 

3. _If the drone is close to the line by some threshold y_<sup>_∗_</sup> _, and at an intermediate pitch angle, the NN output will always be intermediate_ (Listing 2) 

   - _−x_ 5 _− y_<sup>_∗_</sup> _≤ x_ 6 _≤−x_ 5 + _y_<sup>_∗_</sup> _∧−_ 0 _._ 786 _≤ x_ 4 _≤−_ 0 _._ 747 _⇒_ 0 _._ 184 _≤ f_ ( _x_ ) _≤_ 0 _._ 19 (23) 

Our fourth property is more complex, and represents a desireable behaviour not present in the data: 

Neural Network Verification for Gliding Drone Control: A Case Study 

13 

```
droneFarAboveLine:UnnormalisedInputVector->Bool
droneFarAboveLinex=
x!d_y>=-x!d_x+ystar
@property
property1:Bool
property1=forallx.validInputxanddroneFarAboveLinex=>
alsomitrax!e_x>=0.187
```

Listing 1: Property 1 implemented in Vehicle. 

```
intermediatePitch:UnnormalisedInputVector->Bool
intermediatePitchx=
-0.786<=x!d_theta<=-0.747
closeToLine:UnnormalisedInputVector->Bool
closeToLinex=
x!d_y>=-x!d_x-ystarand
x!d_y<=-x!d_x+ystar
@property
property3:Bool
property3=forallx.validInputxandintermediatePitchx
andcloseToLinex=>0.184<=alsomitrax!e_x<=0.19
```

Listing 2: Property 3 implemented in Vehicle. 

4. _If the drone is above and close to the line, pitching down quickly and moving fast, the NN output will always make the drone pitch up_ 

   - _−x_ 5 _≤ x_ 6 _≤−x_ 5 + _y_<sup>_∗_</sup> _∧ x_ 3 _≤−_ 0 _._ 12 _∧ x_ 2 _≤−_ 0 _._ 3 _⇒ f_ ( _x_ ) _≤_ 0 _._ 187 (24) 

In our Vehicle code, _alsomitra_ represent the NN, _validInput_ represents the input space bounded by the training data, and the parameter _ystar_ is defined during runtime. 

### **7.2 Local Robustness Specification** 

Our fifth property is an evaluation of robustness around _ϵ_ -balls with respect to the training dataset, as defined in Sect. 5. A detailed introduction to _ϵ_ -ball robustness for image classification implemented in Vehicle can be found in [10]. Similarly to properties 1-4, we are interested in finding at what threshold _L_ value ( _L_<sup>_∗_</sup> ) does each network pass verification. Similarly to properties 1-4, we expect the verification results for this property to depend on the strictness of _L_ , which we consider as the parameter _L_<sup>_∗_</sup> . However, due to Marabou limitations, this was evaluated with respect to the training dataset, where for each network Property 5 was evaluated for each training point, given fixed _L∗_ and _ϵ_ values. Additionally, the distance between points was computed with _L_<sup>_∞_</sup> and the input distance could not be included in the formula, leading use to use a different robustness definition: 

5. _For any given input point x, the network output f_ ( _x_<sup>_∗_</sup> ) _of any perturbed point x_<sup>_∗_</sup> _within an ϵ-ball around x, will have a distance less than or equal to L_<sup>_∗_</sup> _/ϵ_ 

14 C. Kessler et al. 

_to f_ ( _x_ ) (Listing 3) 



This definition is less strict than our definition of Lipschitz robustness (Eq. 20), since it is effectively equivalent to standard robustness (Eq. 19) where _L_<sup>_∗_</sup> _/ϵ_ = _δ_ . This means that any counterexample to Property 5 will also violate Lipschitz robustness where _L_ = _L_<sup>_∗_</sup> , but not the other way around. Since we train for the stronger definition (Sect. 5), we expect to see improved robustness with regard to this weaker definition. 

In our Vehicle code, parameters _epsilon_ and _Lipschitz_ are defined during runtime, and _n_ is inferred from the training data (provided in idx format [10]). 

```
myList:ListRat
myList=[0,1,2,3,4,5]
boundedByEpsilon:InputVector->Bool
boundedByEpsilonx=foralliinmyList.-epsilon<=x!i-x!i+6<=epsilon
validPerturbation:InputVector->Bool
validPerturbationx=foralliinmyList.x!i==0.0
standardRobustness:InputVector->OutputVector->Bool
standardRobustnessinputoutput=forallpertubation.
letperturbedInput=input-pertubationinvalidPerturbationpertubationand
validInputperturbedInputandboundedByEpsilonperturbedInput=>
(output!e_x-alsomitraperturbedInput!e_x2)<=Lipschitz/epsilonand
alsomitraperturbedInput!e_x2-output!e_x<=Lipschitz/epsilon
```

```
@property
property4:VectorBooln
property4=foreachi.standardRobustness(trainingInputs!i)(trainingOutputs!i)
```

Listing 3: Property 5 implemented in Vehicle. In this case, the states are defined in normalised terms to avoid scaling issues. Instead of calling the network twice to evaluate _f_ ( _x_ ) and _f_ ( _x_<sup>_∗_</sup> ), the NN is doubled in onnx format so that two sets of inputs and outputs can be evaluated at once. 

Neural Network Verification for Gliding Drone Control: A Case Study 

15 

### **7.3 Verification Results** 

**Table 1.** Critical _y_<sup>_∗_</sup> values for properties 1-4 (Sect. 7.1), for naive and adversarially trained NNs (Sect. 5), and for PDT NNs (Sect. 6). For properties 1, 2, and 4, a lower _y_<sup>_∗_</sup> indicates a controller that better adheres to the target trajectory, and the inverse for Property 3. 

|Property|Naive|Adversarial|DL2|Gödel Logic|
|---|---|---|---|---|
|1 (21)|46|30|30|27|
|2 (22)|42|42|42|42|
|3 (22)|Failed|Failed|||
|4 (23)|0|0|0|0|



**Table 2.** Verification success rates (%) of Property 5 (Sect. 7.2) for naive and adversarial NNs (Sect. 5), per _L_<sup>_∗_</sup> values and _ϵ_ , with respect to the training dataset. A higher success rate means that the NN is robust with respect to more of the training data points. As _ϵ_ increases we are increasing the radius for perturbation around each training point, and a decreasing _L_<sup>_∗_</sup> results in a stricter maximum gradient threshold. Empty cells represent properties that timed out before verifying 100 data points. 

||Naive||_L_<sup>_∗_</sup><br>|||
|---|---|---|---|---|---|
|||0_._00001|0_._0001|0_._001|0_._01|
||0.00001|100|100|100|100|
||0.0001|96_._1|100|100|100|
|_ϵ_|0.001|17_._0|98_._5|100|100|
||0.01|-|-|-|100|
||||_L_<sup>_∗_</sup>|||
|A|dversarial|0_._00001|0_._0001|0_._001|0_._01|
||0.00001|100|100|100|100|
||0.0001|99_._7|100|100|100|
|_ϵ_|0.001|23_._9|99_._7|100|100|
||0.01|0|13_._6|92_._5|100|



These results provide interesting insights from an engineering perspective. From Table 1 there is a clear improvement in performance for Property 1 when implementing adversarial training and PDT, suggesting that our approaches have been successful. However, Properties 1 and 2 only succeed with very large 

16 C. Kessler et al. 

values of _y_<sup>_∗_</sup> - our controllers only adhere to a region around the target trajectory so wide as to be useless in real applications. Properties 3 and 4 failed and succeeded, for all _y_<sup>_∗_</sup> values, for all the tested networks, suggesting that they are particularly difficult and easy to verify respectively. Table 2 shows a marginal improvement in robustness performance for our adversarial network, suggesting that our approach has been moderately successful. 

## **8 CORA Implementation** 

### **8.1 Reachability Specification** 

Since our case study is based on the QUAD benchmark from the ARCH competition [25], our reachability specification is defined similarly. The initial set is: 



The reachability goal is for the drone to always be within a distance _y_<sup>_∗_</sup> = 2 of the target trajectory after 20 s. To compute reachability, CORA uses set representations (such as zonotopes [2]) and set operations to over-approximate the continuous time reachable set in discrete time steps. Additional considerations include the initial set representation, time step size, controller type, and reachability algorithm - in our case we use a zonotope, 0.01 s, a NN controller, and conservative linearization. 

### **8.2 Reachability Results** 



<!-- Start of picture text -->
\<br>><br><!-- End of picture text -->



**Fig. 7.** Reachable regions in _x_ 5 and _x_ 6 for naive and adversarial NNs implemented as controllers, from the initial set defined in Eq. 26. The naive NN fails to reach a region bounded by _y_<sup>_∗_</sup> = 2 after 20 s, and the adversarial NN succeeds. 

This result shows significant improvement from the adversarial network compared to the naive (these networks are identical to those from Tables 1 and 2), with the adversarial network adhering much closer to the target trajectory. This would suggest that our adversarial training has been successful, but we note 

Neural Network Verification for Gliding Drone Control: A Case Study 

17 

that whilst these networks are identical in terms of structure, training data, and training epochs, the discrepancy in this plot could be partly due to differences in regression performance. 

## **9 Discussion** 

### **9.1 Limitations and Lessons Learned** 

A use case such as ours requires multiple tools to verify, each with benefits and drawbacks. Although standard methods apply, modifications are needed for proper implementation, such as with robustness training (Sect. 5). Vehicle does not have integration with Python, and our dynamics model and CORA are in Matlab, requiring the use of three programming languages for our case study. Marabou does not support multiple network calls, so a workaround involving doubling the NN (in .onnx format) was required to evaluate robustness in Vehicle (3). Additionally, for our local robustness property, certain Marabou queries timed out after very few data points (Table 2). Correct handling of normalisation was found to be very important, since the implemented robustness training and verification methods require normalisation but CORA does not support normalisation. To evaluate reachability in CORA with normalised networks required another workaround, incorporating the normalisation arithmetic as extra layers in the network. Furthermore, CORA could not evaluate reachability for the full system of equations described in Sect. 3.2, due to their complexity and the size of the starting set. 

Implementing CORA for our application was difficult due to set explosions, where the complexity of the reachability computation would cause an exponential expansion in the set, causing a crash. To reduce the complexity of the equations, the angle of attack definition was simplified ( 1). To solve the initial set size issue, the initial set was divided in _x_ 6, and the resulting subsets combined to a final reachable set. The reachability timesteps still needed to be small (0.01 s) to avoid set explosions; resulting in long computation times for full reachable sets (over 8 hours in some cases). All reachable sets only involve a starting interval in _x_ 6 (starting values are constants in every other dimension), since increasing the starting interval size in multiple dimensions would cause set explosions. These limitations in CORA were found to stem from the complexity of the partial derivative matrices (Jacobian and Hessian), especially due to a large number of nonlinear terms. CORA could be improved significantly by faster approximations of these derivatives, such as with finite differences instead of symbolic methods, but performing such calculations over sets was found to be non-trivial. ReLU activation functions were also found to reduce computation times compared to sigmoid. 

Another consideration for future work is the trade-off between robustness and regression performance for such a controller. For our verification implementations we assume that each NN is capable of controlling the drone relatively well, but that may not always have been the case. A better comparison could be made using NNs with equivalent regression performance on a test set of data, using a 

18 C. Kessler et al. 

coefficient such as _R_<sup>2</sup> . Additionally, the effect of PDT [13] on our system was not fully investigated, partly due to differences in network structure and unresolved issues with the Marabou solver. 

### **9.2 Conclusion** 

In this paper, we introduce a novel case study in the verification of gliding drones (Sect. 4). We developed a two-dimensional model for _Alsomitra_ -inspired drones in Sect. 3, presented an ideal verification formalisation in Sect. 4.1, reduced the formalisation into properties manageable by current tools (Vehicle and CORA, Sects. 7 and 8), and presented verification results in Sects. 7.3 and 8.2. We noted challenges that this class of problems presents, among which are its more complex dynamics, under-defined notion of safe and unsafe states, preference for infinite-time horizon guarantees, and its different learning-as-regression regimes. We have shown that in principle, a combination of existing verification tools (Sect. 2.2) and novel training methods (Sects. 5 and 6) could be effectively adopted in order to enable future inclusion of this class of benchmarks into NNV portfolios. We note, however, that verification tasks like this motivate strongly development of tools that cross the boundaries of ARCH-COMP and VNN-COMP on the one hand, and incorporate these tools more smoothly with machine learning toolboxes on the other hand; cohering perhaps with the general agenda of building more complex programming language interfaces for such more complex verification tasks [8]. Technical problems such as insufficient support for normalisation (as reported here) can make a difference between verification success and failure, yet they are often over-looked in papers and tools that are dedicated to implementing NNV algorithms. Whilst we can define and verify for the behaviours that we want to a certain extent, the state of tools makes it difficult to say exactly how well NNs will fulfil their role - even for our relatively simple (two-dimensional, non-turbulent) drone system. For example, Table 1 suggests that our NNs are only guaranteed to adhere to a very large region around the line, and Fig. 7 shows reachable regions but for which the initial set is relatively small (zero-width in 5 dimensions). All of these issues will likely be found on any comparable regression task with complex dynamic equations. If these limitations can be overcome, it will help enable engineers to develop safe and robust control and modelling methods for technologies that improve people’s lives and reduce our impact on the environment. 

## **References** 

1. Abouheaf, M., Mailhot, N., Gueaieb, W.: An online reinforcement learning wingtracking mechanism for flexible wing aircraft. In: 2019 IEEE International Symposium on Robotic and Sensors Environments (ROSE). pp. 1–7 (2019). `https: //doi.org/10.1109/ROSE.2019.8790425` 

2. Althoff, M., Kochdumper, N., Ladner, T., Wetzlinger, M.: Manual v2025 (2024), `https://tumcps.github.io/CORA/data/archive/manual/Cora2025Manual.pdf` 

Neural Network Verification for Gliding Drone Control: A Case Study 19 

3. Amer, K., Samy, M., Shaker, M., ElHelw, M.: Deep convolutional neural network based autonomous drone navigation. In: Proceedings of the Thirteenth International Conference on Machine Vision. vol. 11605, p. 1160503 (2021). `https: //doi.org/10.1117/12.2587105` 

4. Azuma, A., Okuno, Y.: Flight of a samara, alsomitra macrocarpa. Journal of Theoretical Biology **129** (3), 263–274 (1987). `https://doi.org/https://doi.org/10. 1016/S0022-5193(87)80001-2` 

5. Brix, C., Bak, S., Johnson, T.T., Wu, H.: The 5th international verification of neural networks competition (vnn-comp 2024): Summary and results (2024), `https://www.arxiv.org/pdf/2412.19985` 

6. Casadio, M., Komendantskaya, E., Daggitt, M.L., Kokke, W., Katz, G., Amir, G., Refaeli, I.: Neural network robustness as a verification property: A principled case study. Computer Aided Verification pp. 219–231 (2022) 

7. Certini, D.: The flight of Alsomitra macrocarpa. Phd thesis, University of Edinburgh (February 2023) 

8. Cordeiro, L.C., Daggitt, M.L., Girard-Satabin, J., Isac, O., Johnson, T.T., Katz, G., Komendantskaya, E., Lemesle, A., Manino, E., Šinkarovs, A., Wu, H.: Neural network verification is a programming language challenge (2025), `https://arxiv. org/abs/2501.05867` 

9. Cummins, C., Seale, M., Macente, A., Certini, D., Mastropaolo, E., Viola, I.M., Nakayama, N.: A separated vortex ring underlies the flight of the dandelion. Nature **562** , 414–418 (2018), `https://doi.org/10.1038/s41586-018-0604-2` 

10. Daggitt, M., Kokke, W., Komendantskaya, E., Atkey, B., Arnaboldi, L., Slusarz, N., Casadio, M., Coke, B., Lee, J.: A vehicle tutorial (2024), `https://vehicle-lang. github.io/tutorial/` 

11. ERC: A dandelion-inspired drone for swarm sensing. `https://cordis.europa.eu/ project/id/101001499` 

12. Fischer, M., Balunović, M., Drachsler-Cohen, D., Gehr, T., Zhang, C., Vechev, M.: Dl2: Training and querying neural networks with logic. In: International Conference on Machine Learning (2019) 

13. Flinkow, T., Casadio, M., Kessler, C., Monahan, R., Komendantskaya, E.: A Generalised Framework for Property-Driven Machine Learning (2025), submitted to AI Verification 

14. Flinkow, T., Pearlmutter, B.A., Monahan, R.: Comparing differentiable logics for learning with logical constraints. Science of Computer Programming **244** , 103280 (Sep 2025). `https://doi.org/10.1016/j.scico.2025.103280` 

15. Frehse, G., Althoff, M.: Arch-comp24: Volume information proceedings of the 11th int. workshop on applied verification for continuous and hybrid systems. EPiC Series in Computing **103** (2024), `https://easychair.org/publications/volume/ ARCH-COMP24` 

16. Iyer, V., Gaensbauer, H., Daniel, T.L., Gollakota, S.: Wind dispersal of battery-free wireless devices. Nature **603** , 427–433 (2022) 

17. J. Li, Q. Yang, B.F.Y.S.: Robust state/output-feedback control of coaxial-rotor mavs based on adaptive nn approach (2019), `https://ieeexplore.ieee.org/ document/8715436` 

18. Johnson, K., Arroyos, V., Ferran, A., Villanueva, R., Yin, D., Elberier, T., Aliseda, A., Fuller, S., Iyer, V., Gollakota, S.: Solar-powered shape-changing origami microfliers. Science Robotics **8** (82) (2023), `https://www.science.org/doi/abs/10. 1126/scirobotics.adg4276` 

- 20 C. Kessler et al. 

19. Katz, G., Barrett, C., Dill, D., Julian, K., Kochenderfer, M.: Reluplex: An efficient smt solver for verifying deep neural networks (2017), `https://arxiv.org/abs/ 1702.01135` 

20. Katz, G., Huang, D.A., Ibeling, D., Julian, K., Lazarus, C., Lim, R., Shah, P., Thakoor, S., Wu, H., Zeljić, A., Dill, D.L., Kochenderfer, M.J., Barrett, C.: The marabou framework for verification and analysis of deep neural networks pp. 443– 452 (2019) 

21. Kim, B.H., Li, K., Kim, J.T., Park, Y., Jang, H., Wang, X., Xie, Z., Won, S.M., Yoon, H.J., Lee, G., Jang, W.J., Lee, K.H., Chung, T.S., Jung, Y.H., Heo, S.Y., Lee, Y., Kim, J., Cai, T., Kim, Y., Prasopsukh, P., Yu, Y., Yu, X., Avila, R., Luan, H., Song, H., Zhu, F., Zhao, Y., Chen, L., Han, S.H., Kim, J., Oh, S.J., Lee, H., Lee, C.H., Huang, Y., Chamorro, L.P., Zhang, Y., Rogers, J.A.: Threedimensional electronic microfliers inspired by wind-dispersed seeds. Nature (2021), `https://doi.org/10.1038/s41586-021-03847-y` 

22. Kolter, Z., Madry, A.: Adversarial robustness—theory and practice. NeurIPS 2018 tutorial (2018), available at `https://adversarial-ml-tutorial.org/` 

23. Lemesle, A., Lehmann, J., Le Gall, T.: Neural network verification with pyrat (2024), `https://arxiv.org/abs/2410.23903` 

24. Li, H., Goodwill, T., Jane Wang, Z., Ristroph, L.: Centre of mass location, flight modes, stability and dynamic modelling of gliders. Journal of Fluid Mechanics **937** , A6 (2022). `https://doi.org/10.1017/jfm.2022.89` 

25. Lopez, D.M., Althoff, M., Benet, L., Blab, C., Forets, M., Jia, Y., Johnson, T.T., Kranzl, M., Ladner, T., Linauer, L., Neubauer, P., Neubauer, S.A., Schilling, C., Zhang, H., Zhong, X.: Arch-comp24 category report: Artificial intelligence and neural network control systems (ainncs) for continuous and hybrid systems plants. EPiC Series in Computing **103** , 64–121 (2024), `https://easychair.org/ publications/paper/WsgX` 

26. Lumini, M.: Pherodrone1.0: An innovative inflatable uav’s concept, inspired by zanonia macrocarpa’s samara flying-wing and to insect’s sensillae, designed for the biological control of harmful insects in pa (precision agriculture). Bionics and Sustainable Design (2022) 

27. Müller, M.N., Singh, G., Balunovic, M., Makarchuk, G., Ruoss, A., Serre, F., Baader, M., D Cohen, D., Gehr, T., Hoffmann, A., Maurer, J., Mirman, M., Müller, C., Püschel, M., Tsankov, P., Vechev, M.: Eran (2025), `https://github. com/eth-sri/eran` 

28. Oshima, K., Kuribara, K., Sato, T.: Flex-snn: Spiking neural network on flexible substrate. IEEE Sensors Letters **7** (5), 1–4 (2023) 

29. Platzer, A.: Logical Foundations of Cyber-Physical Systems. Springer, Cham (2018). `https://doi.org/10.1007/978-3-319-63588-0` 

30. Qamar, S., Khan, S.H., Arshad, M.A., Qamar, M., Gwak, J., Khan, A.: Autonomous drone swarm navigation and multitarget tracking with island policybased optimization framework. IEEE Access **10** , 91073–91091 (2022). `https: //doi.org/10.1109/ACCESS.2022.3202208` 

31. Richter, D.J., Calix, R.A., Kim, K.: A review of reinforcement learning for fixedwing aircraft control tasks. IEEE Access **12** , 103026–103048 (2024). `https://doi. org/10.1109/ACCESS.2024.3433540` 

32. Singaraju, S.A., Weller, D.D., Gspann, T.S., Aghassi-Hagmann, J., Tahoori, M.B.: Artificial neurons on flexible substrates: A fully printed approach for neuromorphic sensing. Sensors **22** (2022) 

Neural Network Verification for Gliding Drone Control: A Case Study 21 

33. Wada, D., Araujo-Estrada, S.A., Windsor, S.: Unmanned aerial vehicle pitch control using deep reinforcement learning with discrete actions in wind tunnel test. Aerospace **8** , 18 (2021). `https://doi.org/10.3390/aerospace8010018` 

34. Wang, Y., Zhou, W., Fan, J., Wang, Z., Li, J., Chen, X., Huang, C., Li, W., Zhu, Q.: Polar-express: Efficient and precise formal reachability analysis of neural-network controlled systems. IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems **43** (3), 994–1007 (2024). `https://doi.org/10.1109/TCAD. 2023.3331215` 

35. Wiesemüller, F., Meng, Z., Hu, Y., Farinha, A., Govdeli, Y., Nguyen, P.H., Nyström, G., Kovač, M.: Transient bio-inspired gliders with embodied humidity responsive actuators for environmental sensing. Frontiers in Robotics and AI (2023) 

36. WMO: Global Demonstration Campaign for Evaluating the Use of Uncrewed Aircraft Systems in Operational Meteorology: White Paper. Tech. rep. (2023) 

37. Zhang, H., Weng, T.W., Chen, P.Y., Hsieh, C.J., Daniel, L.: Efficient neural network robustness certification with general activation functions. Advances in Neural Information Processing Systems **31** , 4939–4948 (2018), `https://arxiv.org/pdf/ 1811.00866.pdf` 

## **10 Appendix** 

### **10.1 Equations** 

22 C. Kessler et al. 

|**Table 3.** 2D quasi-steady equations for falling pl<br>When integrated over time, the force coefficients <br>the equations of motion (13-18). For simplicity t<br>within the region _α ∈_[_−π/_2_,_0].<br>**Constant(s)**<br>_ℓ_, _m_<br>_ρf_<br>_α_0, _δ_<br>_C_1<br>_L, C_2<br>_L, C_0<br>_D, C_1<br>_D, Cπ/_2<br>_D_<br>_, C_0<br>_CP , C_1<br>_CP , C_2<br>_CP , CR_<br>_a, b_<br>No. **Variable**<br>1<br>CoM displacement, _ℓ_CM_/ℓ_or _ex_<br>2<br>Moment of Inertia, _I_ [kgm2]<br>3<br>Angle of attack, _α_ (p15)<br>4<br>Selection function, _f_(_α_) (5.2)<br>5<br>Lift coefficient, _C_L(_α_) (5.1, 5.3)<br>6<br>Drag coefficient, _C_D(_α_) (5.4, 5.5)<br>7<br>Center of pressure, _ℓ_CP(_α_) (5.6, 5.7)<br>8<br>Translational lift force, _L_T (4.10)<br>9<br>Rotational lift force, _L_R (4.11)<br>10<br>Drag force, _D_ (4.13)<br>11<br>Torque from transl. forces, _τ_T (4.14)<br>12<br>Aerodynamic rot. resistance, _τ_R (4.15)<br>**13** Fixed frame _x_ velocity, ˙_x_ (4.4)<br>**14** Fixed frame _y_ velocity, ˙_y_ (4.5)<br>**15** Angular velocity, ˙_θ_ (4.6)<br>**16** Platewise _x′_ acceleration, ˙_vx′_ (4.7)<br>**17** Platewise _y′_ acceleration, ˙_vy′_ (4.8)<br>**18** Angular acceleration, ˙_ω_ (4.9)|
|---|
|ates with a displaced centre of mass [24], with six system variables (_x, y, θ, vx′, vy′, ω_).<br>are derived from the angle of attack (5-7), the forces are calculated (8-12), followed by<br>he plate is assumed to be infinitesimally thin, and to always have an angle of attack<br>**Definition(s)**<br>**Value(s)**<br>Plate length [m] and mass [kg]<br>0.07, 3.175e-04<br>Fluid Density [kg_/_m3]<br>1.225<br>Critical _α_ at stall, stall transition smoothness [°] 14, 6<br>System-specific aerodynamic coefficients<br>0.23857, 2.8529, 0.36893,<br>5.1822, 0.80751, 0.10598,<br>4.9368, 1.4996, 1.73<br>Elliptical semi axes [m]<br>0.03375, 5e-04<br>**PDE expression**<br>Defined by controller, in the range [0_._181_,_0_._193]<br>_I_ = (_m_(_a_2+_b_2)_/_(_ρfℓ_4)) + 1_/_32 + (_ℓ_CM_/ℓ_)2<br>tan_α_= (_vy′ −ωℓ_CM)_/vx′ ≈vy′/vx′_<br>_f_ = (1_−_tanh((_α −α_0)_/δ_))_/_2<br>_−C_L =_f_(_|α|_)_C_1<br>Lsin(_|α|_) + (1_−f_(_|α|_))_C_2<br>Lsin(2_|α|_)<br>_C_D =_f_(_|α|_)(_C_0<br>D+_C_1<br>Dsin2(_|α|_)) + (1_−f_(_|α|_))_Cπ/_2<br>D<br>sin2(_|α|_)<br>_ℓ_CP_/ℓ_=_f_(_|α|_)(_C_0<br>CP _−C_1<br>CP_α_2)+_C_2<br>CP[1_−f_(_|α|_)](1_−|α| /_(_π/_2))<br>_L_T = 1<br>2_ρfℓC_L<br>~~�~~<br>_vx′_2 + (_vy′ −ωℓ_CM)2 (_vy′ −ωℓ_CM_, vx′_)<br>_L_R =_−_1<br>2_ρfℓ_2_C_R_ω_ (_vy′ −ωℓ_CM_, vx′_)<br>_D_ =_−_1<br>2_ρfℓC_D<br>~~�~~<br>_vx′_2 +(_vy′ −ωℓ_CM)2 (_vx′, vy′ −ωℓ_CM)<br>_τ_T =_−_1<br>2_ρfℓ_<br>~~�~~<br>_vx′_2 + (_vy′ −ωℓ_CM) [_C_L_vx′_ +_C_D(_vy′ −ωℓ_CM)] (_ℓ_CP_−ℓ_CM)<br> _τ_R =_−_<br>1<br>128_ρfℓ_4_Cπ/_2<br>D<br>_ω |ω|_<br>��<br>2_ℓ_CM<br>_ℓ_<br>+ 1<br>�4<br>_±_<br>�<br>2_ℓ_CM<br>_ℓ_<br>_−_1<br>�4�<br>˙_x_=_vx′_cos_θ −vy′_sin_θ_<br>˙_y_ =_vx′_sin_θ_+_vy′_cos_θ_<br>˙_θ_ =_ω_<br>_m_ ˙<br>_vx′_ =<br>�<br>_m_+_πρfℓ_2_/_4<br>�<br>_ωvy′ −_(_πρfℓ_2_/_4)_ω_2_ℓCM_ +_Lx′_<br>_T_ +_Lx′_<br>_R_ +_Dx′ −m′g_sin_θ_<br>�<br>_m_+_πρfℓ_2_/_4<br>�<br>˙_vy′_ =_−mωvx′_ + (_πρfℓ_2_/_4) ˙_ωℓCM_ +_Ly′_<br>_T_ +_Ly′_<br>_R_ +_Dy′ −m′g_cos_θ_<br>_I_˙_ω_ =_τT_ +_τR_|



